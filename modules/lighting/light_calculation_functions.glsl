// Reusable Light Calculation Lighting Functions
// Automatically extracted from lighting-related shaders

// Function 1
vec3 lightDirection()
{
    return -normalize(lightPosition());
}

// Function 2
vec2 AOandFakeAreaLights(vec3 pos, vec3 n)
{
	vec4 res = vec4(0.0);
    
	for( int i=0; i<3; i++ )
	{
		vec3 aopos = pos + n*0.3*float(i);
		vec4 d = distfunc(aopos);
		res += d;
	}
    
    float ao = clamp(res.w, 0.0, 1.0);
    float light = 1.0 - clamp(res.z*0.3, 0.0, 1.0);
    
	return vec2(ao, light * ao);   
}

// Function 3
void updateLights()
{
	sphereRad		= cos( iTime * 0.3 ) * 0.025 + 0.05;
	spherePos		= vec3( sin( iTime * 0.25 ), abs( cos( iTime ) * 0.25 ) + sphereRad, 0.0 );
	
	tubeRad			= sin( iTime * 0.1 ) * 0.005 + 0.01;
	vec3 tubePos	= vec3( 0.0, sin( iTime * 0.3 ) * 0.1 + 0.2, cos( iTime * 0.25 ) );	

	vec3 tubeVec	= rotPitch(rotYaw(vec3(0,0,0.2), iTime*-1.5 ), cos( iTime*0.5 ) * 0.3 );
	
	tubeStart		= tubePos + tubeVec;
	tubeEnd			= tubePos - tubeVec;
}

// Function 4
BRDFOutput ApplyDirectionalLight(in DirectionalLight light, in Material material, in vec3 point, in vec3 normal, in vec3 eye, in bool castShadow)
{
    vec3 lightDirection = normalize(-light.mDirection);
    BRDFOutput returnValue = GetBRDFOutput(point, normal, eye, lightDirection, light.mColor, 1.0, material);
                 
    // Cast a ray to check for shadows
    float shadow = 1.0;
#if SHADOWS_ENABLED
    if(castShadow)
    {
    	vec3 shadowRayDirection = (gSceneInfo.x == MONTE_CARLO_MODE) ? RandomDirectionAroundRange(float(iFrame), 0.7, lightDirection) : lightDirection;
    	Ray shadowRay = Ray(point + (EPSILON * normal), shadowRayDirection);
    	IntersectionPoint lightIntersection = CheckSceneForIntersection(shadowRay);
    	shadow = IsIntersectionValid(lightIntersection) ? 0.0 : shadow; // Determine if we hit an object and are in a shadow region 
    }
#endif // SHADOWS_ENABLED
    returnValue.mLighting *= shadow;
    
	return returnValue;
}

// Function 5
void DoTheLighting(RayIntersection ri, out vec4 c)
{
    float attByDst;
    float NoL, specAngle;
    float shadow, ao;
    float d2l = MAX_DISTANCE;
    vec3  diffuse, specular;
    vec3  L, halfVec;
    vec4  ambient;
    
    if (ri.shape.texID != 0)
    {
        ri.shape.color.rgb *= TriplanarTextureMapping(ri.pos,
                                                     ri.shape.normal.xyz,
                                                     ri.shape.texID);
    }
    
	for (int i=0; i<NUM_LIGHTS; i++)
    {
        if (scene.lights[i].type == DIRECTIONAL)
        {
            L 		 = -scene.lights[i].dir;
            attByDst = 1.0;
        }
        else if (scene.lights[i].type == POINT)
        {
            vec3  p2l = scene.lights[i].pos - ri.pos;
            d2l = length(p2l); 
            if (d2l > scene.lights[i].range) continue;
            attByDst = (scene.lights[i].range - d2l) / scene.lights[i].range;
            L = normalize(p2l);
        }

        // BLINN-PHONG
        // Diffuse component
        NoL      = clamp(dot(L, ri.shape.normal.xyz), .0, 1.0);
        diffuse += NoL * attByDst *
            	   scene.lights[i].color * scene.lights[i].intensity;
        
        // Specular component
        if (NoL >= .0 && ri.shape.glossy > .0)
        {
            halfVec    = normalize(-ri.ray.d + L);
            specAngle  = clamp(dot(ri.shape.normal.rgb, halfVec), .0, 1.0);
            specular  += pow(specAngle, ri.shape.glossy*512.) * attByDst *
                		 scene.lights[i].color * scene.lights[i].intensity;
        }

        shadow += ComputeShadow(ri.pos, ri.shape.normal.xyz, L, d2l);
    }
    // Ambient Occlusion
    ao = ComputeAO(ri.pos, ri.shape.normal.xyz);
    
    //ambient = AMBIENT_LIGHT;
    ambient = texture(iChannel0, reflect(ri.ray.d, ri.shape.normal.xyz),
                       1./ri.shape.glossy);

    // Combine all the illumination components
    c  = ri.shape.color * vec4(diffuse, 1);
    c *= shadow * ao;
    c += ri.shape.color * ambient;
	c += vec4(specular, 0);
    
    // DEBUG: Normals
    //c = ri.shape.normal;
    // DEBUG: Ambient Occlusion
    //c = vec4(ao,ao*.5,0,1);
    
    // NOTE: Applying the fog here keeps the sky gradient,
    // but makes the horizon look too sharp
	//ApplyFog(c.rgb, ri.distance);
}

// Function 6
vec2 sampleLight(vec3 rp, vec3 n)
{
    vec2 energy = vec2(0);
    vec3 lightdir = normalize(normalize(vec3(.2, 4., 1.)) +
                              (vec3(rand(), rand(), rand()) * 2. - 1.) * .015);
    vec3 n2, uvw;
    float t = traceScene(rp, lightdir, n2, uvw).x;

    vec3 lrp = rp + lightdir * t;

    // Directional 'sky' lighting.
    if((lrp.y > .999 && abs(lrp.x- -.3) < .6 && abs(lrp.z - .1) < .8))
        energy += vec2(1.5, .8).yx * max(0., dot(n, lightdir)) * 2.;

    vec3 lo = vec3(.7, .8, .1), ls = vec3(1, 0, 0) * .2, lt = vec3(0, 0, 1) * .2;
    vec3 ln = normalize(cross(ls, lt));
    
    int light_sample_count = 2;
    
    // Parallelogram local lightsource.
    for(int j = 0; j < light_sample_count; ++j)
    {
        float lu = rand() * 2. - 1., lv = rand() * 2. - 1.;
        vec3 lp = lo + ls * lu + lt * lv, n2;
        float ld = dot(normalize(lp - rp), n), ld2 = dot(normalize(rp - lp), ln);
        if(ld > 0. && ld2 > 0. && traceSceneShadow(rp + n * 1e-4, lp - rp))
            energy += vec2(1.5, .5) *
            	(1. / dot(rp - lp, rp - lp) * ld * ld2) / float(light_sample_count);
    }

    return energy;
}

// Function 7
vec3 OrenNayarLightModel(vec3 rd, vec3 ld, vec3 n, float albedo)
{
	vec3 col = vec3(0.);
	float RDdotN = dot(-rd, n);
	float NdotLD = dot(n, ld);
    float aRDN = acos(RDdotN);
	float aNLD = acos(NdotLD);
	float mu = 5.; // roughness
	float A = 1.-.5*mu*mu/(mu*mu+0.57);
    float B = .45*mu*mu/(mu*mu+0.09);
	float alpha = max(aRDN, aNLD);
	float beta = min(aRDN, aNLD);
	float e0 = 4.8;
	col = vec3(albedo / mPi) * cos(aNLD) * (A + ( B * max(0.,cos(aRDN - aNLD)) * sin(alpha) * tan(beta)))*e0;
	return col;
}

// Function 8
vec4 lightInf(Ray ray,float t,in vec3 prevColor, in float counter,in float omega)
{
    Ray lineRay = Ray(vec3(-10.0, 10.0 ,10.0), 0.0, vec3(1.0, 0.0, 0.0), 0.0);
    vec3 rayPV = ray.P + ray.V*t;
    vec3 lightPos= lightPath(lineRay,10.0,counter);
    vec3 normalizeLPR = normalize(lightPos - rayPV); //The vector from the point to the light source
    Ray shadowRay = Ray(rayPV, 0.05, normalizeLPR, distance(rayPV, lightPos));
    SphereTraceDesc params = SphereTraceDesc(0.01, 128);
    TraceResult traceRes  = enhancedSphereTrace(shadowRay, params, omega);
    
    vec3 normSurf = normal(ray.P + ray.V * t);  
  
    int i = 1;
    
    if(prevColor == vec3(0, 0, 0))
    	i =  2;
    else if(prevColor == vec3(1, 0, 0))
    	i =  0;

    
    if(bool(traceRes.flags & 1) && bool(i & 1))
        return vec4(vec3(0.5, 0.5, 0.5)*max(dot(normSurf, normalizeLPR), 0.0) / counter + prevColor, 1.0);
    else if(bool(traceRes.flags & 1) && !bool(i & 1))
        return vec4(vec3(0.2, 1, 0.2) * max( dot(normSurf, normalizeLPR), 0.0)/counter, 1.0);
    else if(!bool(traceRes.flags & 1) && bool(i & 1))
        return vec4(prevColor, 1.0);
    else if(bool(traceRes.flags & 2) || bool(i & 2))
        return vec4(0.0, 0.0, 0.0, 1.0);
    else
    	return vec4(1.0, 0.0, 0.0, 1.0);

}

// Function 9
vec3 phongContribForLight(vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye,
                          vec3 lightPos, vec3 lightIntensity) {
    lightPos = eye;
    vec3 N = estimateNormal(p);
    vec3 L = normalize(lightPos - p);
    vec3 V = normalize(eye - p);
    vec3 R = normalize(reflect(-L, N));
    
    float dotLN = dot(L, N);
    float dotRV = dot(R, V);
    
    if (dotLN < 0.0) {
        // Light not visible from this point on the surface
        return vec3(0.0, 0.0, 0.0);
    } 
    
    if (dotRV < 0.0) {
        // Light reflection in opposite direction as viewer, apply only diffuse
        // component
        return lightIntensity * (k_d * dotLN);
    }
    return lightIntensity * (k_d * dotLN + k_s * pow(dotRV, alpha));
}

// Function 10
float diffuseLight (vec3 p, vec3 n, vec3 r){
    return dot(n, -r);
}

// Function 11
vec3 light (in ray r, in vec3 color) {
	return color / (dot (r.p, r.p) + color);
}

// Function 12
vec3 sampleLightBRDF( in vec3 hitOrigin, in vec3 hitNormal, in vec3 rayDir, in Material material  )
{
    vec3 brdf = vec3( 0 );
    vec3 s = vec3( 0 );

    Light light;
    light.id = 3.0;
    light.emission = LIGHT1_EM;

    vec3 l0 = vec3( 2, 2, 4 ) - hitOrigin;

    float cos_a_max = sqrt(1. - clamp(0.5 * 0.5 / dot(l0, l0), 0., 1.));
    float cosa = mix(cos_a_max, 1., random());
    vec3 l = jitter(l0, 2.*PI*random(), sqrt(1. - cosa*cosa), cosa);

#if (PATH == 1)
    vec3 lightHit = castRay( hitOrigin, l, 0.001, 100.0 );
    if ( lightHit.z == light.id )
#else
    s += softshadow( hitOrigin, normalize(l0) );
#endif
    {
        float roughness = 1.0 - material.smoothness * material.smoothness;
        float metallic = material.metallic;

        float omega = 2. * PI * (1. - cos_a_max);
        brdf += ((light.emission * clamp(ggx( hitNormal, rayDir, l, roughness, metallic),0.,1.) * omega) / PI);
    }

    light.id = 4.0;

    l0 = vec3( -4, 1.5, 4 ) - hitOrigin;

    cos_a_max = sqrt(1. - clamp(0.5 * 0.5 / dot(l0, l0), 0., 1.));
    cosa = mix(cos_a_max, 1., random());
    l = jitter(l0, 2.*PI*random(), sqrt(1. - cosa*cosa), cosa);

#if (PATH == 1)
    lightHit = castRay( hitOrigin, l, 0.001, 100.0 );
    if ( lightHit.z == light.id )
#else
    s += softshadow( hitOrigin, normalize(l0) );
#endif
    {
        float roughness = 1.0 - material.smoothness * material.smoothness;
        float metallic = material.metallic;

        float omega = 2. * PI * (1. - cos_a_max);
        brdf += ((light.emission * clamp(ggx( hitNormal, rayDir, l, roughness, metallic),0.,1.) * omega) / PI);
    }

#if (PATH == 0)
    brdf *= clamp( s, 0., 1. );
#endif

    return brdf;
}

// Function 13
vec3 SampleLight(Object light, int lightId, int ignoreObjId, vec3 P, vec3 N, vec2 s, bool sphericalLightIsTextured)
{
	vec3 V;
    vec3 L;
    float inversePDF_d;
    if (IsQuad(light)) {
        V = QuadLocalToWorld(s*2.0 - vec2(1.0), light) - P;
        L = normalize(V);
		float distSqr = dot(V, V);
        inversePDF_d = GetQuadArea(light)*max(0.0, -dot(light.quadNormal, L))/distSqr;
    } else {
        // http://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources.html
        vec3 pointToLight = light.pos - P;
		float radiusSqr = light.radius*light.radius;
        float sinThetaMaxSqr = radiusSqr/dot(pointToLight, pointToLight);
        float cosThetaMax = sqrt(1.0 - sinThetaMaxSqr);
        float cosTheta = cosThetaMax + (1.0 - cosThetaMax)*s.y;
        float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
		if (sphericalLightIsTextured) {
			float dc = length(pointToLight);
			float ds = dc*cosTheta - sqrt(max(0.0, radiusSqr - dc*dc*sinTheta*sinTheta));
			float cosAlpha = (radiusSqr + dc*dc - ds*ds)/(2.0*dc*light.radius);
			float sinAlpha = sqrt(max(0.0, 1.0 - cosAlpha*cosAlpha));
			V = light.pos + light.radius*SampleHemisphere(-normalize(pointToLight), sinAlpha, cosAlpha, s.x) - P;
			L = normalize(V);
		} else {
			V = vec3(0);
			L = SampleHemisphere(normalize(pointToLight), sinTheta, cosTheta, s.x);
		}
		inversePDF_d = 2.0*PI*(1.0 - cosThetaMax);
    }
	inversePDF_d *= max(0.0, dot(N, L))/PI;
    if (inversePDF_d > 0.0 && IntersectScene(Ray(P, L), ignoreObjId) == lightId)
		return SampleLightColor(P + V, light)*inversePDF_d;
	else
		return vec3(0);
}

// Function 14
vec3 lightPosition()
{
    vec3 dir = vec3(0.0, 2.0, 0.0);
    
    dir = rotX(dir, -0.15 * TAU);
    dir = rotY(dir, time * 0.1 * TAU);
    
    return dir;
}

// Function 15
mat4 GetDirectionalLightMatrixInverse ()
{
    vec4 rotAxisAngle = GetDirectionalLightRotationAxisAngle();
    vec3 trans = GetDirectionalLightSourcePosition();
	mat4 rot = rotationAxisAngle(rotAxisAngle.xyz, rotAxisAngle.w );
	mat4 tra = translate( trans.x, trans.y, trans.z );
	return tra * rot; 
}

// Function 16
LightSample sampleLight(in vec3 p, in Light light)
{
    vec3 difference = light.p - p;
    float distance = length(difference);
    vec3 direction = difference / distance;
    
    float sinTheta = light.r / distance;
	float cosTheta = sqrt(1.0 - sinTheta * sinTheta);
    
    LightSample result;
    
    vec3 hemi = squareToUniformSphereCap(rand2(), cosTheta);
    result.pdf = squareToUniformSphereCapPdf(cosTheta);
    
    vec3 s = normalize(cross(direction, vec3(0.433, 0.433, 0.433)));
    vec3 t = cross(direction, s);
    
    result.d = (direction * hemi.z + s * hemi.x + t * hemi.y);
    return result;
}

// Function 17
vec3 traceLight( in vec3 from, in vec3 norm, in uvec3 seed ) {
    
    vec3 pos = vec3(0);
    vec4 diff = vec4(0);
    vec3 dummyNorm = vec3(0);
    
    // create a random dir in a hemisphere
    vec3 rand = hash(seed);
    float dirTemp1 = 2.0*PI*rand.x;
    float dirTemp2 = sqrt(1.0-rand.y*rand.y);
    vec3 dir = vec3(
        cos(dirTemp1)*dirTemp2,
        sin(dirTemp1)*dirTemp2,
        rand.y);
    dir.y = abs(dir.y);
    
    // pick the sun more often (priority sampling)
    const float sunContrib = colorSun.a*2.0*PI*(1.0 - sunCosAngle);
    const float ambientContrib = colorAmbient.a*2.0*PI;
    const float groundContrib = colorGround.a*2.0*PI;
    const float sumContrib = sunContrib+ambientContrib+groundContrib;
    
    float a = sunContrib / sumContrib;
    float b = a + ambientContrib / sumContrib;
    
    if (rand.z < a) {
        const vec3 sunDirTan = normalize(cross(sunDir, vec3(0, 0, 1)));
        const vec3 sunDirCoTan = cross(sunDir, sunDirTan);
        float rot = 2.0*PI*rand.x;
        float the = acos(1.0 - rand.y*(1.0 - cos(sunAngle)));
        float sinThe = sin(the);
        dir = sunDirTan*sinThe*cos(rot) + sunDirCoTan*sinThe*sin(rot) - sunDir*cos(the);
    } else if (rand.z < b) {
        dir.z = abs(dir.z);
    } else {
        dir.z = -abs(dir.z);
    }
    
    if (trace(from, dir, false, pos, dummyNorm, diff)) {
        vec3 back = getBackground(dir);
        vec3 color = back.rgb * diff.rgb * (1.0 - diff.a);
        float l = dot(norm, norm) > 0.0 ? max(0.0, dot(dir, norm)) : 1.0;
        return color*l*sumContrib;
    } else {
        return vec3(0);
    }
    
}

// Function 18
vec3 lighting(Ray r, vec3 dif, vec3 norm, float sp, bool shd){
    vec3 col = dif * light.amb;
    
    Ray scheck = r;
    scheck.dir = -light.dir;
    if(shd && norm.y >= 0.9/*magic?*/ && danObject(scheck).x >=0.0) { return col;}
    
    float dp = clamp(dot(norm, -light.dir), 0., 1.);
    col += dif * dp * light.col;
    
    if(sp > 0.0){
        vec3 rfn = reflect(-light.dir, norm);
        dp = clamp(dot(r.dir, rfn), 0., 1.);
        col += light.col * pow(dp, sp);
    }
    return col;
}

// Function 19
float hmd_flight_path_marker( vec2 coord )
{
    float result = 0.;
    vec3 localv = g_vehicle.modes.x == VS_HMD_ORB ?
        g_vehicle.orbitv * g_planet.B * g_game.camframe :
    	g_vehicle.localv * g_game.camframe;
    if( g_vrmode )
        localv *= g_vrframe;
    if( dot( localv, localv ) >= .25e-6 )
    {
        vec3 v = localv;
        float sz = hmd_symbol_border( v, HMD_BORDER_SYM );
        mat2 I = mat2( g_textscale.x, 0, 0, g_textscale.y );
    	vec2 p = ( coord - project3d( v, g_game.camzoom ) ) * g_textscale;
        vec2 a = vec2( +4, 0 );
        vec2 b = vec2( +9, 0 );
        vec2 c = vec2( 0, +4 );
        float shape = 0.;
        if( Linfinity( p ) < 10. )
        {
        	shape = max( shape, aaa_ring( I, p, sz * 8., 1. ) );
            shape = max( shape, aaa_hline( I, p, -sz * b, sz * 5., 1. ) );
            shape = max( shape, aaa_hline( I, p, +sz * a, sz * 5., 1. ) );
            shape = max( shape, aaa_vline( I, p, +sz * c, sz * 4., 1. ) );
        }
		if( localv.x < 0. &&
            abs( localv.y ) < -HMD_BORDER_SYM.x * localv.x &&
            abs( localv.z ) < -HMD_BORDER_SYM.y * localv.x )
		{
        	p = ( coord - project3d( localv, g_game.camzoom ) ) * g_textscale;
            if( Linfinity( p ) < 10. )
            {
            	shape = max( shape, aaa_ring( I, p, 8., 1. ) );
            	shape = max( shape, aaa_hline( I, p, -a, 8., 1. ) );
            	shape = max( shape, aaa_vline( I, p, -c, 8., 1. ) );
            }
        }
        result += shape * sz;
    }
    return result;
}

// Function 20
vec3 scene_object_lighting( vec3 albedo, vec3 N, vec3 L, vec3 V, vec3 Z, vec3 F,
                            vec3 skyZ, vec3 skyL, vec3 skyR, vec3 ground )
{
    float mu_0 = mu_stretch( dot( N, L ), .01 );
    float mu = mu_stretch( dot( N, V ), .01 );
    float cosi = dot( N, Z );
    float cosp = dot( L, V );
    float cost = dot( normalize( reject( N, Z ) ), normalize( reject( L, Z ) ) );
    vec3 kd = lunar_lambert( albedo, mu, mu_0 );
    float kl = phase_curve( cosp );
    vec3 E = F * mu_0;
    //*
    vec3 sky = mix( mix( skyR, skyL, .5 + .5 * cost ), skyZ, cosi * .3333 + .6667 );
    return E * kd * kl + albedo * mix( ground, sky, cosi * .5 + .5 );
    /*/
    float cosi2 = cosi * cosi;
    vec3 skyH = ( skyL + skyR ) / 2.;
    vec3 skyJ = ( skyL - skyR ) / 2.;
    vec3 sky = skyZ / 8. * ( 2.6667 + cosi * ( 3.5 + cosi2 * ( -0.3333 + cosi2 * ( -0.5 + cosi2 ) ) ) ) +
               skyH / 8. * ( 1.3333 + cosi * ( 0.5 + cosi2 * ( +0.3333 + cosi2 * ( +0.5 + cosi2 ) ) ) ) +
               skyJ * cost / ( 105. * PI ) * ( 30. - cosi2 * ( 6. + cosi2 * ( 8. + cosi2 * 16. ) ) );
    return E * kd * kl + albedo * ( sky + ground * ( 1. - cosi ) / 2. );
    //*/
}

// Function 21
vec3 volumetricLight(vec3 p, vec3 ro, vec3 rd, vec2 uv)
{
#ifdef VOLUMETRIC_ACTIVE
    vec3 col = vec3(0.0);
    float val = 0.0;
    
   	p -= rd * noise(9090.0*uv) * 0.6;
    vec3 s = -rd * 2.2 / float(VOLUMETRIC_STEPS);
    
    for (int i = 0; i < VOLUMETRIC_STEPS; i++)
    {
        float v = getVisibility(p, light_pos, 250.0) * .015;
        p += s;
        float t = exp(p.z - 3.0);
        val += v * t;
    }  
    
    return vec3(min(val, .8));
#else
    return vec3(0.0);
#endif
}

// Function 22
float MapStreeLight(  vec3 p)
{
  float d= fCylinder(p-vec3(0.31, -3.5, 0.), 0.7, 0.01);
  d=fOpPipe(d, fCylinder(p-vec3(.31, -4., 0.), 0.7, 3.0), .05);   
  d=min(d, fCylinderH(p-vec3(.98, -6.14, 0.), 0.05, 2.4));        
  d=fOpUnionChamfer(d, fCylinderH(p-vec3(.98, -8., 0.), 0.1, 1.0), 0.12);  
  d=min(d, sdSphere(p-vec3(-0.05, -3.4, 0.), 0.2));  
  d=min(d, sdSphere(p-vec3(-0.05, -3.75, 0.), 0.4));        
  d=max(d, -sdSphere(p-vec3(-.05, -3.9, 0.), 0.45)); 

  return d;
}

// Function 23
bool isLightVisible(in vec3 ro, in int id)
{
	float dist = length(ld[id].pos - ro);
	vec3 rd = normalize(ld[id].pos - ro);
	float tmin;
	vec3 nor;
	int oid = iScene(ro, rd, tmin, nor);
	if (oid == -1 || tmin > dist)
	{
		return true;
	}
	else
	{
		return false;
	}
}

// Function 24
bool getLightPulse()
{
    return texture(iChannel2, vec2(2.5, 0.5) / iResolution.xy).y < 0.5;
}

// Function 25
vec3 getLightPosition(int l
){return 14.*vec3(cos(iTime*1.61)
                 ,sin(iTime*sqrt(7.)*.4)*2.+3.
                 ,sin(iTime)
                 );}

// Function 26
vec3 getPhysicalLighting(in Material mat, in PointLight light, in vec3 position, in vec3 normal, in Ray camRay, in int seed)
{
    vec3 v = normalize(camRay.origin - position);
    vec3 wi = normalize(light.position - position);
    vec3 h = normalize(wi + v);
    
    float cosTheta = max(dot(normal, wi), 0.0);
    
    float attenuation = calculateAttenuation(light.position, position);
    vec3 radiance = light.color * attenuation;
    
    vec3 F0 = vec3(0.04); 
	F0      = mix(F0, mat.albedo, mat.metalness);
	vec3 F  = fresnelSchlick(max(dot(h, v), 0.0), F0);
    
    float NDF = DistributionGGX(normal, h, mat.roughness);       
	float G   = GeometrySmith(normal, v, wi, mat.roughness);
    
    vec3 numerator    = NDF * G * F;
	float denominator = 4.0 * max(dot(normal, v), 0.0) * max(dot(normal, wi), 0.0);
	vec3 specular     = numerator / max(denominator, 0.001);
    
    vec3 kS = F;
    vec3 kD = vec3(1) - kS;
    kD *= 1.0 - mat.metalness;
    
    vec3 indirectDiffuse = vec3(0); 
    for (int i = 0; i < PASSES; i++) {
        indirectDiffuse += indirectDiffuseCast(camRay, seed);
    }
    indirectDiffuse /= float(PASSES);
    
    return indirectDiffuse * kD + specular * radiance * cosTheta;
}

// Function 27
vec3 sampleLightType( const in LightInfo light, const in SurfaceInteraction interaction, out vec3 wi, out float lightPdf, out float visibility, float seed) {
    if( !light.enabled )
        return vec3(0.);
    
    if( light.type == LIGHT_TYPE_SPHERE ) {
        vec3 L = lightSample(light, interaction, wi, lightPdf, seed);
        visibility = visibilityTest(interaction.point + wi * .01, wi);
        return L;
    }
    else if( light.type == LIGHT_TYPE_SUN ) {
        vec3 L = sampleSun(light, interaction, wi, lightPdf, seed);
        visibility = visibilityTestSun(interaction.point + wi * .01, wi);
        return L;
    }
    else {
        return vec3(0.);
    }
}

// Function 28
void initLights() {
    #if (NUM_LIGHTS != 0)
    //lights[0] = light(0., 0.5, normalize(dirLight));
    #endif
}

// Function 29
float lightPdf(const in vec4 light, const in SurfaceInteraction interaction) {
    float sinThetaMax2 = light.w * light.w / distanceSq(light.xyz, interaction.point);
    float cosThetaMax = sqrt(max(EPSILON, 1. - sinThetaMax2));
    return 1. / (TWO_PI * (1. - cosThetaMax));
}

// Function 30
vec4 getSpotLightOne(vec2 uv, vec2 center, float intensity, vec3 color){
	float ratio = iResolution.x/iResolution.y; 
    uv.x *= ratio;
    center.x *= ratio;
    float dist = intensity/sqrt(distance(uv, center));
    return vec4(color * dist, dist);
}

// Function 31
SG GetLightSG()
{
    float t = iTime * .5;
    float height = .15;
    vec3 p = vec3(cos(t), height, sin(t));
    
	SG light;
	light.Axis = normalize(p);
    light.Sharpness = 10.0;
    light.Amplitude = pow(vec3(2.5, 1.5, 1.05), vec3(2.2)) * .35;
    return light;
}

// Function 32
vec3 lightsTexture(vec2 uv){
    vec3 col = vec3(0.);
    vec2 auv = abs(uv-vec2(0.5));
    float f = 1.0-smoothstep(0.2,0.35,max(auv.x,auv.y));
    f = pow(f,2.);
    //col = mix(col, vec3(1.2,1.2,0.9), f);
    col = mix(col, vec3(0.95,0.95,1.), f);
    return col;
}

// Function 33
vec3 tower_lightpos(vec2 cell_coords, vec3 tower_color)
{
    // no movement in center column
    float mask = step(REALLY_SMALL_NUMBER, abs(dot(cell_coords, vec2(1.))));
    vec3 light_color = mask * tower_color;
    return vec3(0., mod(5. * g_time + 200. * tower_color.r, 110.) - 100., 0.);

    // return vec3(0., mod(5. * g_time + 10. * (cell_coords.x + 4. * cell_coords.y), 110.) - 100., 0.);
    // return vec3(0., mod(10. * g_time + 2. * cell_coords.x, 200.) - 100., 0.);
}

// Function 34
vec4 atm_skylight_sample( TrnSampler ts, sampler2D ch, vec3 x )
{
    vec2 res = vec2( textureSize( ch, 0 ) );
    vec2 aspect = vec2( res.y / res.x, 1 );
    vec2 uv = ts_uv( ts, x ) / 2. * aspect + vec2( 0, .5 );
    return textureLod( ch, uv, 0. );
}

// Function 35
vec3 directionalLight(vec3 normalizedNormal, vec3 position, vec3 lightPos, vec3 viewPosition, vec3 color, float shadow)
{
    vec3 lightColor = vec3(1.);
    // ambient
    vec3 ambient = lightColor * 0.15 * color;
    
    vec3 lightDirection = normalize(lightPos-position);
    vec3 viewDirection = normalize(viewPosition-position);
	vec3 halfwayDirection = normalize(lightDirection + viewDirection);
    
    // diffuse
    float diffuseIntensity = max(dot(normalizedNormal, lightDirection),0.0);
    vec3 diffuse = diffuseIntensity * lightColor * color;
    
    // specular
    float specularStrength = .85;
    float specularIntensity = pow(max(dot(normalizedNormal, halfwayDirection),0.0),32.);
    //// going glossy on the specular (no color multiplication)
    vec3 specular = specularStrength * specularIntensity * lightColor;						
    return (ambient + (diffuse + specular)*(1. - shadow));

    
}

// Function 36
BRDFOutput CalculateLighting(in Material material, in vec3 point, in vec3 normal, in vec3 eye)
{
    BRDFOutput returnValue;
    
    DirectionalLight directionalLight = DirectionalLight(vec3(0.0, -1.0, 0.0), vec3(1.0));;    
    returnValue = ApplyDirectionalLight(directionalLight, material, point, normal, eye, true);

    directionalLight.mDirection = normalize(vec3(15.0 * sin(gTimeValue), -8.0, 15.0 * cos(gTimeValue)));
    returnValue = AddBRDFOutput(returnValue, ApplyDirectionalLight(directionalLight, material, point, normal, eye, true));
    
    directionalLight.mDirection = normalize(vec3(-15.0 * cos(-gTimeValue), -8.0, 15.0 * sin(-gTimeValue)));
    returnValue = AddBRDFOutput(returnValue, ApplyDirectionalLight(directionalLight, material, point, normal, eye, true));
    
    return returnValue;
}

// Function 37
vec3 Lighting(vec3 normal, vec3 rd) {
    vec3 diffuse = clamp(dot(normal, -moonDir), 0., 1.) * moonCol;
    vec3 specular = pow(clamp(dot(normalize(-(rd + moonDir)), normal), 0., 1.), 8.) 
        * moonCol;
    return diffuse + specular;
}

// Function 38
vec3 scene_lighting_ocean( vec3 albedo, vec3 Z, vec3 N, vec3 M, vec3 L, vec3 V, vec3 F,
                           float a, vec3 sky,
                           vec4 refl, float extra_T )
{
#if WITH_ILLUM_TEST
    float mu0 = max( 0., dot( N, L ) );
    return F * mu0 + refl.xyz;
#else
    // variation of the KSK microfacet model
    vec3 L_refract = normalize( -simple_refract( -L, Z ) );
    float mu0_refract = max( 0., dot( N, L_refract ) ) * max( 0., dot( L, Z ) );
    float mu0 = max( 0., dot( M, L ) );
    float mu = max( 0., dot( M, V ) );
    vec3 H = normalize( L + V );
    float cosxi = max( 0., dot( M, H ) );
    float cospsi = max( .0625, dot( L, H ) );
    float fr_mu = refl.w;
    float fr_psi = fresnel_schlick( .02, cospsi );
    float kd = ( 1. - fr_mu );
    float ks = extra_T * NDFdisk( cosxi, a, .5 * square( 7487. / 321226. ) ) / ( 4. * cospsi * cospsi );
    return F * mix( mu0_refract * albedo * kd, mu0 * vec3( ks ), fr_psi ) +
        albedo * sky * ( 1. - fr_mu ) + refl.xyz;
#endif
}

// Function 39
vec3 SampleLightMIS_h(Object light, int lightId, int ignoreObjId, vec3 P, vec3 N, vec2 s, float N_d, float N_h)
{
	vec3 L = SampleHemisphereCosineWeighted(N, s);
	float t;
	Object unused;
	if (IntersectScene(Ray(P, L), ignoreObjId, t, unused) == lightId) {
		vec3 V = L*t;
		float inversePDF_d;
		if (IsQuad(light)) {
			float distSqr = t*t; // same as dot(V, V)
			inversePDF_d = GetQuadArea(light)*max(0.0, -dot(light.quadNormal, L))/distSqr;
		} else {
			vec3 pointToLight = light.pos - P;
			float radiusSqr = light.radius*light.radius;
			float sinThetaMaxSqr = radiusSqr/dot(pointToLight, pointToLight);
			float cosThetaMax = sqrt(1.0 - sinThetaMaxSqr);
			inversePDF_d = 2.0*PI*(1.0 - cosThetaMax);
		}
		inversePDF_d *= max(0.0, dot(N, L))/PI;
		if (inversePDF_d > 0.0) {
			float PDF_d = 1.0/inversePDF_d;
			float PDF_h = 1.0;
		#if MIS_USE_POWER
			float b = MIS_power_b;
			return SampleLightColor(P + V, light)*pow(N_h*PDF_h, b - 1.0)/(pow(N_d*PDF_d, b) + pow(N_h*PDF_h, b));
		#else
			return SampleLightColor(P + V, light)/(N_d*PDF_d + N_h*PDF_h);
		#endif
		}
	}
	return vec3(0);
}

// Function 40
vec3 SampleLightMIS_d(Object light, int lightId, int ignoreObjId, vec3 P, vec3 N, vec2 s, float N_d, float N_h, bool sphericalLightIsTextured)
{
	vec3 V;
    vec3 L;
    float inversePDF_d;
    if (IsQuad(light)) {
        V = QuadLocalToWorld(s*2.0 - vec2(1.0), light) - P;
	    L = normalize(V);
		float distSqr = dot(V, V);
        inversePDF_d = GetQuadArea(light)*max(0.0, -dot(light.quadNormal, L))/distSqr;
    } else {
        // http://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources.html
        vec3 pointToLight = light.pos - P;
		float radiusSqr = light.radius*light.radius;
        float sinThetaMaxSqr = radiusSqr/dot(pointToLight, pointToLight);
        float cosThetaMax = sqrt(1.0 - sinThetaMaxSqr);
        float cosTheta = cosThetaMax + (1.0 - cosThetaMax)*s.y;
        float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
		if (sphericalLightIsTextured) {
			float dc = length(pointToLight);
			float ds = dc*cosTheta - sqrt(max(0.0, radiusSqr - dc*dc*sinTheta*sinTheta));
			float cosAlpha = (radiusSqr + dc*dc - ds*ds)/(2.0*dc*light.radius);
			float sinAlpha = sqrt(max(0.0, 1.0 - cosAlpha*cosAlpha));
			V = light.pos + light.radius*SampleHemisphere(-normalize(pointToLight), sinAlpha, cosAlpha, s.x) - P;
			L = normalize(V);
		} else {
			V = vec3(0);
			L = SampleHemisphere(normalize(pointToLight), sinTheta, cosTheta, s.x);
		}
        inversePDF_d = 2.0*PI*(1.0 - cosThetaMax);
	}
	inversePDF_d *= max(0.0, dot(N, L))/PI;
    if (inversePDF_d > 0.0 && IntersectScene(Ray(P, L), ignoreObjId) == lightId) {
		float PDF_d = 1.0/inversePDF_d;
		float PDF_h = 1.0;
	#if MIS_USE_POWER
		float b = MIS_power_b;
		return SampleLightColor(P + V, light)*pow(N_d*PDF_d, b - 1.0)/(pow(N_d*PDF_d, b) + pow(N_h*PDF_h, b));
	#else
		return SampleLightColor(P + V, light)/(N_d*PDF_d + N_h*PDF_h);
	#endif
	}
	return vec3(0);
}

// Function 41
vec3 sampleLightType( const in LightInfo light, const in SurfaceInteraction interaction, out vec3 wi, out float lightPdf, out float visibility, float seed, const in MaterialInfo material) {
    if( !light.enabled )
        return vec3(0.);
    
    if( light.type == LIGHT_TYPE_SPHERE ) {
        vec3 L = lightSample(light, interaction, wi, lightPdf, seed, material);
        vec3 shadowRayDir =normalize(light.position - interaction.point);
        visibility = visibilityTest(interaction.point + shadowRayDir * .01, shadowRayDir);
        return L;
    }
    else if( light.type == LIGHT_TYPE_SUN ) {
        vec3 L = sampleSun(light, interaction, wi, lightPdf, seed);
        visibility = visibilityTestSun(interaction.point + wi * .01, wi);
        return L;
    }
    else {
        return vec3(0.);
    }
}

// Function 42
void RenderAnimatedLights(vec2 uv, inout vec2 I, inout float AFlag) {
    //Animated lights
    float uvlen=length((uv-vec2(0.25-pow(sin(iTime*0.1),2.)*0.04,0.78+0.055*(1.-exp(-iTime*0.15))))*ASPECT);
        if (uvlen<IRES.y*2.) { I.y=0.4+0.2*(uvlen/(IRES.y*2.)); AFlag=1.; } //Helicopter
}

// Function 43
void constructLightPath(inout float seed) {
    vec3 ro = normalize( hash3(seed)-vec3(0.5) );
    vec3 rd = randomHemisphereDirection( ro, seed );
    ro = lightSphere.xyz + ro*0.5;
    vec3 color = LIGHTCOLOR;
    
    lpNodes[0].position = ro;
    lpNodes[0].color = color;
    lpNodes[0].normal = rd;
    
    for( int i=1; i<LIGHTPATHLENGTH; ++i ) {
        lpNodes[i].position = lpNodes[i].color = lpNodes[i].normal = vec3(0.);
    }
    
    for( int i=1; i<LIGHTPATHLENGTH; i++ ) {
		vec3 normal;
        vec2 res = intersect( ro, rd, normal );
        if( res.y > -0.5 && res.y < 4. ) {
            ro = ro + rd*res.x;
            color *= calcColor( res.y );
            lpNodes[i].position = ro;
            lpNodes[i].color = color;
            lpNodes[i].normal = normal;

            rd = cosWeightedRandomHemisphereDirection( normal, seed );
        } else break;
    }
}

// Function 44
void initLightSphere( float time ) {
	lightSphere = vec4( 3.0+2.*sin(time),2.8+2.*sin(time*0.9),3.0+4.*cos(time*0.7), .1 );
}

// Function 45
vec3 kali_set_with_light(in vec3 pos, in vec3 param)
{
    vec4 p = vec4(pos, 1.);
    vec3 d = vec3(100.);
    for (int i=0; i<9; ++i)
    {
        p = abs(p) / dot(p.xyz,p.xyz);
        vec3 s = p.xyz/p.w;
        d = min(d, s);
        if (i == 3)
            light = vec4(.5+.5*sin(pos.xzx*vec3(8,9,19)), 
                         length(s.xz)-0.003);
        p.xyz -= param;
    }
    return d;
}

// Function 46
vec3 light( float t )
{
    return vec3(2.,3.5*sin(4.*t),2.);
}

// Function 47
vec3 areaLights( vec3 pos, vec3 nor, vec3 rd )
{
	float noise		=  texture( iChannel1, pos.xz ).x * 0.5;
	noise			+= texture( iChannel1, pos.xz * 0.5 ).y;
	noise			+= texture( iChannel1, pos.xz * 0.25 ).z * 2.0;
	noise			+= texture( iChannel1, pos.xz * 0.125 ).w * 4.0;
	
	vec3 albedo		= pow( texture( iChannel0, pos.xz ).xyz, vec3( 2.2 ) );
	albedo			= mix( albedo, albedo * 1.3, noise * 0.35 - 1.0 );
	float roughness = 0.7 - clamp( 0.5 - dot( albedo, albedo ), 0.05, 0.95 );
	float f0		= 0.3;
	
	#ifdef DISABLE_ALBEDO
	albedo			= vec3(0.1);
	#endif
	
	#ifdef DISABLE_ROUGHNESS
	roughness		= 0.05;
	#endif
	
	vec3 v			= -normalize( rd );
	float NoV		= clamp( dot( nor, v ), 0.0, 1.0 );
	vec3 r			= reflect( -v, nor );
	
	float NdotLSphere;
	float specSph	= sphereLight( pos, nor, v, r, f0, roughness, NoV, NdotLSphere );
	
	float NdotLTube;
	float specTube	= tubeLight( pos, nor, v, r, f0, roughness, NoV, NdotLTube );
	
	vec3 color		= albedo * 0.3183 * ( NdotLSphere + NdotLTube ) + specSph + specTube;
	return pow( color, vec3( 1.0 / 2.2 ) );
}

// Function 48
void lightLogic() {

    vec2 uv = iMouse.xy - iResolution.xy / 2.;
    uv /= iResolution.y;
    
    if(iMouse.z > 0.) {

        Ray ray;
        ray.origin = vec3(0.,.35,-3.);
        ray.direction = normalize(vec3(uv.x,uv.y-.3,1.));
        
        RaycastHit scene = raycastScene(ray);
        
        if(scene.type > -1) {
         
            light1 = scene.point;
            return;
            
        }
        
    }
     
    light1 = vec3(-5., 5., -6.);
    
}

// Function 49
vec4 lighting(vec3 n, vec3 rayDir, vec3 reflectDir, vec3 pos)
{
    vec3 light = vec3(0.0, 0.0, 2.0 + iTime * speed);
    vec3 lightVec = light - pos;
	vec3 lightDir = normalize(lightVec);
    float atten = clamp(1.0 - length(lightVec)*0.1, 0.0, 1.0);
    float spec = pow(max(0.0, dot(reflectDir, lightDir)), 10.0);
    float rim = (1.0 - max(0.0, dot(-n, rayDir)));

    return vec4(spec*atten*lightColor2 + rim*0.2, rim); 
}

// Function 50
void calcPointLight(inout Light light, Surface surface, vec3 cameraPos) {
  	float d = distance(light.position, surface.position);
  	vec3 k = vec3(.06, .08, .09);
  	light.attenuation = 1. / (k.x + (k.y * d) + (k.z * d * d));

  	// point light
  	vec3 lightDir = normalize(light.position - surface.position);
  	// diffuse
  	float diffuseCoef = max(0., dot(surface.normal, lightDir));
  	vec3 diffuse = diffuseCoef * light.color * light.intensity * light.attenuation;
  	// specular
  	float specularCoef = getSpecular(
        surface,
        lightDir,
        diffuseCoef,
        cameraPos
    );
  	vec3 specular = vec3(specularCoef * light.attenuation * light.color * light.intensity); 
    
  	light.diffuse = diffuse * softShadow(surface.position, normalize(light.position), .1, 10., 10.);
  	light.specular = specular;
}

// Function 51
vec3 sampleLight( const in vec3 ro ) {
    lowp vec3 n = randomSphereDirection() * lightSphere.w;
    return lightSphere.xyz + n;
}

// Function 52
float directLighting( in vec3 pos, in vec3 nor )
{

    vec3 ww = lig;
    vec3 uu = normalize( cross(ww, vec3(0.0,1.0,0.0)) );
    vec3 vv =          ( cross(uu,ww) );


    float shadowIntensity = softshadow( pos+0.001*nor, lig, 10.0 );

    vec3 toLight = rlight - pos;
    float att = smoothstep( 0.985, 0.997, dot(normalize(toLight),lig) );

    vec3 pp = pos - ww*dot(pos,ww);
    vec2 uv = vec2( dot(pp,uu), dot(pp,vv) );
    float pat = smoothstep( -0.5, 0.5, sin(10.0*uv.y) );

    return pat * att * shadowIntensity;
}

// Function 53
vec3 lighting(Surface surface, vec3 cameraPos, vec3 cameraDir) {
  	vec3 position = surface.position;

  	vec3 color = vec3(0.);
  	vec3 normal = surface.normal;

  	Light directionalLight;
  	directionalLight.position = vec3(.5, 1., .5);
  	directionalLight.intensity = .7;
  	directionalLight.color = vec3(1., 1., 1.);
  	calcDirectionalLight(directionalLight, surface, cameraPos);
    
  	Light pointLight;
  	pointLight.position = vec3(sin(iTime) * 2., 2., 1.);
  	pointLight.intensity = .3;
  	pointLight.color = vec3(1., 1., 1.);
  	calcPointLight(pointLight, surface, cameraPos);
    
  	vec3 diffuse = directionalLight.diffuse + pointLight.diffuse;
  	vec3 specular = directionalLight.specular + pointLight.specular;
    
    // calc ambient
	float occ = ambientOcculusion(surface.position, surface.normal);
  	float amb = clamp(.5 + .5 * surface.normal.y, 0., 1.);
  	vec3 ambient = surface.baseColor * amb * occ * vec3(0., .08, .1);  
    
  	color =
        surface.emissiveColor +
        surface.baseColor * diffuse +
        surface.specularColor * specular +
        ambient;  
  
  	return color;
}

// Function 54
vec3 GetSceneLight(float specLevel, vec3 normal, RayHit rayHit, vec3 rayDir, vec3 origin)
{        
  vec3 reflectDir = reflect( rayDir, normal );

  float amb = clamp( 0.5+0.5*normal.y, 0.0, 1.0 );
  float dif = clamp( dot( normal, sunPos ), 0.0, 1.0 );
  float bac = clamp( dot( normal, normalize(vec3(-sunPos.x, 0.0, -sunPos.z))), 0.0, 1.0 )*clamp( 1.0-rayHit.hitPos.y, 0.0, 1.0);
  float fre = pow( clamp(1.0+dot(normal, rayDir), 0.0, 1.0), 2.0 );
  specLevel*= pow(clamp( dot( reflectDir, sunPos ), 0.0, 1.0 ), 16.0);

  float skylight = smoothstep( -0.1, 0.1, reflectDir.y );
  vec3 shadowPos = origin+((rayDir*rayHit.depth)*0.99);  
  dif *= SoftShadow( shadowPos, sunPos);
  skylight *=SoftShadow(shadowPos, reflectDir);

  vec3 lightTot = vec3(0.0);

    
    
  lightTot += 1.30*dif*vec3(1.00, 0.80, 0.55);
  lightTot += 0.50*skylight*vec3(0.40, 0.60, 1.00);
      lightTot += 1.20*specLevel*vec3(0.9, 0.8, 0.7)*dif;
  lightTot += 0.50*bac*vec3(0.25, 0.25, 0.25);
  lightTot += 0.25*fre*vec3(1.00, 1.00, 1.00);
  return lightTot +(0.40*amb*vec3(0.40, 0.60, 1.00));
}

// Function 55
vec2 lightMap( vec3 pos ){

    
   float dist =length( pos - lightPos ) - .3;
    
    return vec2( dist , 4. );
    
}

// Function 56
float lightIntensity(in vec3 normal, in vec3 light, in vec2 uv) {
    vec3 to_light = light - vec3(uv, 0);
    float dist2 = dot(to_light, to_light)/62500.;
    // Light on the surface of pool. Generaly would be due to particles floating on it.
    float intensity = dot(normal, normalize(to_light)) / dist2;
    
    // Specular reflection of the light source on water.
    vec3 reflected = reflect(vec3(0, 0, -1), normal);
    float dist = distance(dot(to_light, reflected) * reflected, to_light);
    intensity += smoothstep(100., 30., dist) * .9;
    return intensity;
}

// Function 57
vec3 scene_lighting_terrain( vec3 albedo, vec3 N, vec3 L, vec3 V, vec3 Z, vec3 F,
                             vec3 sky, vec2 shadow )
{
#if WITH_ILLUM_TEST
    float mu0 = max( 0., dot( N, L ) );
    return F * mu0 + sky;
#else
    float mu_0 = mu_stretch( dot( N, L ), .01 );
    float mu = mu_stretch( dot( N, V ), .01 );
    float cosi = dot( N, Z );
    float cosp = dot( L, V );
    vec3 kd = lunar_lambert( albedo, mu, mu_0 );
    float kl = phase_curve( cosp );
    float kj = cosi * .5 + .5;
    vec3 E = F * mu_0 * shadow.x;
    vec3 backbounce = .5 * albedo * F * shadow.y * mu_stretch( dot( N, -L ), .125 )
        * mu_stretch( dot( L, Z ), .005 );
    return E * kd * kl + albedo * ( sky * kj + backbounce );
#endif
}

// Function 58
void calcDirectionalLight(inout Light light, Surface surface, vec3 cameraPos) {
    light.position = normalize(light.position);
    light.attenuation = 1.;
    
    // diffuse
  	float diffuseCoef = max(0., dot(surface.normal, normalize(light.position)));
  	vec3 diffuse = diffuseCoef * light.attenuation * light.color * light.intensity;
  	// specular
  	float specularCoef = getSpecular(
        //light,
        surface,
        light.position,
        diffuseCoef,
        cameraPos
    );
  	vec3 specular = vec3(specularCoef * light.attenuation * light.color * light.intensity);  

  	light.diffuse = diffuse * softShadow(surface.position, normalize(light.position), .1, 10., 10.);
    light.specular = specular;
}

// Function 59
vec3 Lighting_Ring(
    in vec3  toView,
    in vec3  pos,
    in vec3  norm,
    in vec3  albedo,
    in float intensity,
    in float roughness)
{
    Light_Ring light;
    light.position    = LightPos;
    light.attenuation = LightAttenuation;
    light.normal      = Rotate(LightNormal, LightYaw, LightPitch);
    light.radius      = LightRadius;
    
    // Get the closest point on the circle circumference on the plane 
    // defined bythe light's position and normal. This point is then used
    // for the standard lighting calculations.
    
    vec3 pointOnLight = PointOnCircle(pos, light.position, light.normal, light.radius);
    vec3 toLight      = pointOnLight - pos;
    vec3 toLightN     = normalize(toLight);
    
    float lightCos    = dot(-toLightN, light.normal);
          lightCos    = (LightDirection == LightDirectionBi) ? abs(lightCos) : max(0.0, lightCos);
    
    float lightDist   = length(toLight);
    float attenuation = Attenuation(lightDist, light.attenuation.x, light.attenuation.y, light.attenuation.z);
    
    vec3 reflVector = reflect(-toView, norm);
    
    vec3 dirLight = vec3(intensity) * lightCos * attenuation;
    vec3 spcLight = vec3(0.2) * pow(max(0.0, dot(reflVector, toLightN)), roughness) * lightCos;
    
    return (albedo * dirLight) + spcLight;
}

// Function 60
vec3 calculateLighting(vec3 position, vec3 normal, vec2 uv, light lights[LIGHTS], vec3 camera)
{
    //convert from tangent space to world space
    {
        mat3 conversion;
        conversion[0] = normalize(cross(vec3(0, 0, 1), normal));
        conversion[1] = normalize(cross(normal, conversion[0]));
        conversion[2] = normal;
        {
            vec3 normmap = texture(iChannel0, uv).rgb * 2.0 - 1.0;
            normmap = normalize(normmap);
            normal = normmap * conversion;
        }
    }
    vec3 total = vec3(0.0);
    for(int i = 0; i < LIGHTS; ++i)
    {
        
        vec3 to_light = lights[i].position - position;
        vec3 incoming = normalize(position - camera);
        total +=
            (
                max(dot(normal, normalize(to_light)), 0.0) + 
                pow(max(dot(normal, normalize((to_light + incoming) * 0.5)), 0.0), EXPONENT)
            ) * lights[i].color.a / length(to_light) * lights[i].color.rgb;
    }
    #if TEXTURED == 1
    return total * texture(iChannel0, uv).rgb * 0.5;
    #else
    return total * 0.5;
    #endif
}

// Function 61
vec3 SampleLightColor(vec3 P, Object light)
{
	vec3 color = light.emissive*light_intensity;
#if LIGHT_TEXTURED
	if (light.quadLightTexture != 0U) {
		vec3 V = P - light.pos;
		float vx = dot(V, light.quadBasisX); // note quadBasis is divided by extent
		float vy = dot(V, light.quadBasisY);
		vec2 uv = vec2(vx, -vy)*0.5 + vec2(0.5); // [0..1]
		color *= textureLod(sampler2D(light.quadLightTexture), uv, light_texture_LOD).rgb;
	}
#endif // LIGHT_TEXTURED
	//if (IsSphere(light)) {
	//	vec3 V = normalize(P - light.pos);
	//	vec3 A = abs(V);
	//	color *= mix(vec3(0.1), V*0.5 + vec3(0.5), pow(max(max(A.x, A.y), A.z), 8.0));
	//}
	return color;
}

// Function 62
mat4 GetDirectionalLightMatrix ()
{
    vec4 rotAxisAngle = GetDirectionalLightRotationAxisAngle();
    vec3 trans = GetDirectionalLightSourcePosition();
	mat4 rot = rotationAxisAngle(rotAxisAngle.xyz, -rotAxisAngle.w );
	mat4 tra = translate( -trans.x, -trans.y, -trans.z );
	return rot * tra;     
}

// Function 63
vec4 NormalPointLight(int val, in IntersectResult ir){
    vec3 v = lights[val].xyz-ir.p;
    return vec4(1,1,1,0)*max(0.,dot(ir.n,normalize(v)))*pow(max(1.-min(1.,length(v)/lights[val].w),0.),3.);
}

// Function 64
float getLightInterp(samplerCube cubeSampler, vec3 lmn) {
    vec3 flmn = floor(lmn);

    float d000 = LIGHT( flmn );
    float d001 = LIGHT( flmn + vec3(0.0, 0.0, 1.0) );
    float d010 = LIGHT( flmn + vec3(0.0, 1.0, 0.0) );
    float d011 = LIGHT( flmn + vec3(0.0, 1.0, 1.0) );
    float d100 = LIGHT( flmn + vec3(1.0, 0.0, 0.0) );
    float d101 = LIGHT( flmn + vec3(1.0, 0.0, 1.0) );
    float d110 = LIGHT( flmn + vec3(1.0, 1.0, 0.0) );
    float d111 = LIGHT( flmn + vec3(1.0, 1.0, 1.0) );

    vec3 t = lmn - flmn;
    return mix(
        mix(mix(d000, d100, t.x), mix(d010, d110, t.x), t.y),
        mix(mix(d001, d101, t.x), mix(d011, d111, t.x), t.y),
        t.z
    );
}

// Function 65
bool raymarch_to_light(vec3 ray_start, vec3 ray_dir, float maxDist, float maxY, out float dist, out vec3 p, out int iterations, out float light_intensity) {
    dist = 0.; 
    float minStep = 1.0;
    light_intensity = 1.0;
	float mapDist;
    for (int i = 1; i <= MAX_RAYMARCH_ITER_SHADOWS; i++) {
        p = ray_start + ray_dir * dist;
        mapDist = mapBlocks(p, ray_dir).y;
        if (mapDist < MIN_RAYMARCH_DELTA) {
            iterations = i;
            return true;
        }
		light_intensity = min(light_intensity, SOFT_SHADOWS_FACTOR * mapDist / dist);
		dist += max(mapDist, minStep);
        if (dist >= maxDist || p.y > maxY) { break; }
    }
    return false;
}

// Function 66
vec3 sampleIndirectLight(vec3 pos,vec3 normal
){vec3 dir
 ;vec3 abso=vec3(1.),light=vec3(0.),dc,ec
 ;for(int i=0;i<Bounces;i++
 ){dir=getCosineWeightedSample(normal)
  ;if(!trace(pos,dir,normal))return light+abso*background(dir)
  ;sdf(pos,dc,ec)
  ;light+=abso*(ec+dc*directLight(pos,normal))
  ;abso*=dc;}
 ;return light;}

// Function 67
vec3 LightTower(vec3 RP, vec3 D, vec3 cRET) { //60, 230, 60
    //Using the same light-positions and types for all 4 towers
    vec3 RET=cRET; vec3 LRET;
    LRET=LightRing(vec3(30.,200.,30.),7.,-0.75,RP,D,5.,1.,0.5,1); if (LRET.z<RET.z) RET=LRET; //Överst
        LRET=LightRing(vec3(30.,190.,30.),9.,RP,D,8.,0.9,0.6,1); if (LRET.z<RET.z) RET=LRET;
        LRET=LightRing(vec3(30.,180.,30.),9.,RP,D,8.,0.9,0.6,1); if (LRET.z<RET.z) RET=LRET;
    LRET=LightRing(vec3(30.,80.,30.),18.,RP,D,16.,1.5,0.5,0); if (LRET.z<RET.z) RET=LRET; //Plattform
            LRET=LightRing(vec3(30.,85.,30.),8.,RP,D,16.,2.,0.6,0); if (LRET.z<RET.z) RET=LRET;
        LRET=LightRing(vec3(30.,50.,30.),17.,RP,D,11.,1.5,0.55,0); if (LRET.z<RET.z) RET=LRET;
        LRET=LightRing(vec3(30.,30.,30.),17.,RP,D,11.,1.5,0.55,0); if (LRET.z<RET.z) RET=LRET;
    return RET;
}

// Function 68
vec3 LightSample(vec3 p,float x1,float x2,out vec3 wo,out float dist,out float pdf){
	vec3 v0v1 = quads[0].v1 - quads[0].v0;
    vec3 v0v3 = quads[0].v3 - quads[0].v0;
    float width  = length(v0v1);
    float height = length(v0v3);
    vec3 O = quads[0].v0 + v0v1*x1 + v0v3*x2;
    wo = O - p;
    dist = length(wo);
    wo = normalize(wo);
    float costhe = dot(-wo,quads[0].normal);
    pdf = PDF_Area2Angle(1./(width*height),dist,clamp(costhe,0.00001,1.));
    return costhe>0. ? GetLightIntensity(): vec3(0.);
}

// Function 69
vec3 CalcLighting(Light light, Intersection i, vec3 origin)
{
	vec3 n = i.n;
	vec3 p = i.p;
	vec3 l = normalize(light.p-p);
	vec3 v = normalize(origin-p);
	vec3 h = normalize(l+v);
	float NdotL = saturate(dot(n,l));
	float NdotH = saturate(dot(n,h));
	vec3 diffuse = NdotL*i.diffuse;
	vec3 spec = pow(NdotH,8.0) * i.specular;
	float distA = 1.0-saturate(length(light.p-p)/light.radius);
	vec3 color;
	color = (diffuse+spec) * distA * light.color;
	
	float shadow = 1.0;
	Ray shadowRay;
	shadowRay.o = i.p;
	float lightDist = length(light.p-i.p);
	shadowRay.dir = (light.p-i.p)/lightDist;
	Intersection shadowI = SceneIntersection(shadowRay);
	if(shadowI.dist < lightDist)
	{
		shadow = 0.0;
	}
	color *= shadow;
	
	return color;
}

// Function 70
float tubeLight( vec3 pos, vec3 N, vec3 V, vec3 r, float f0, float roughness, float NoV, out float NoL )
{
	vec3 L0			= tubeStart - pos;
	vec3 L1			= tubeEnd - pos;
	float distL0	= length( L0 );
	float distL1	= length( L1 );
	
	float NoL0		= dot( L0, N ) / ( 2.0 * distL0 );
	float NoL1		= dot( L1, N ) / ( 2.0 * distL1 );
	NoL				= ( 2.0 * clamp( NoL0 + NoL1, 0.0, 1.0 ) ) 
					/ ( distL0 * distL1 + dot( L0, L1 ) + 2.0 );
	
	vec3 Ld			= L1 - L0;
	float RoL0		= dot( r, L0 );
	float RoLd		= dot( r, Ld );
	float L0oLd 	= dot( L0, Ld );
	float distLd	= length( Ld );
	float t			= ( RoL0 * RoLd - L0oLd ) 
					/ ( distLd * distLd - RoLd * RoLd );
	
	vec3 closestPoint	= L0 + Ld * clamp( t, 0.0, 1.0 );
	vec3 centerToRay	= dot( closestPoint, r ) * r - closestPoint;
	closestPoint		= closestPoint + centerToRay * clamp( tubeRad / length( centerToRay ), 0.0, 1.0 );
	vec3 l				= normalize( closestPoint );
	vec3 h				= normalize( V + l );
	
	float HoN		= clamp( dot( h, N ), 0.0, 1.0 );
	float HoV		= dot( h, V );
	
	float distLight	= length( closestPoint );
	float alpha		= roughness * roughness;
	float alphaPrime	= clamp( tubeRad / ( distLight * 2.0 ) + alpha, 0.0, 1.0 );
	
	float specD		= specTrowbridgeReitz( HoN, alpha, alphaPrime );
	float specF		= fresSchlickSmith( HoV, f0 );
	float specV		= visSchlickSmithMod( NoL, NoV, roughness );
	
	return specD * specF * specV * NoL;
}

// Function 71
vec3 ComputeLight(vec3 P, vec3 N, vec3 LD, vec2 rand, float iTime) {
    HIT LHIT; mat3 NM; vec3 Sample; vec3 Color=vec3(0.);
    #ifdef DEF_SUN
    //Direct Light
    float LDot=dot(LD,N);
    if (LDot>0.) {
        NM=TBN(LD);
        Sample=normalize(RandSample(rand)*NM*SunConeRatio+LD);
        if (!TraceRay(P,Sample,LHIT,64.,iTime))
            Color.xyz+=vec3(0.9,0.7,0.3)*2.*LDot;
    }
    #endif
    //Diffuse
    NM=TBN(N);
    Sample=RandSample(rand)*NM;
    if (TraceRay(P,Sample,LHIT,64.,iTime)) {
        if (LHIT.Mat==2.) Color+=LHIT.C; else {
            //Direct Light
            #ifdef DEF_SUN
            LDot=dot(LD,LHIT.N);
    		if (LDot>0.) {
                NM=TBN(LD);
                Sample=normalize(RandSample(rand)*NM*SunConeRatio+LD);
                if (!TraceRay(LHIT.P+LHIT.N*0.05,Sample,LHIT,64.,iTime))
                    Color.xyz+=vec3(0.9,0.7,0.3)*2.*LDot*LHIT.C;
            }
            #endif
        }
    } else
        Color.xyz+=(Sample.y*0.5+0.5)*vec3(0.1,0.3,0.5);
    //Return
    return Color;
}

// Function 72
vec3 LightBoxShadow (in vec3 position) {
    
    float shade = 1.0;

    // calculate whether or not the position is inside of the light box
    vec2 uv;
	vec3 localPos;    
    if (GetMode() >= 1.0)
    {
        // get position in light space and get uv
        localPos = WorldSpaceToDirectionalLightSpace(position);
    	uv = localPos.xz;
        uv.x *= -1.0;
        
        // apply scaling of uv over distance to fake projection
        uv /= (1.0 + localPos.y * directionalLightUVDistanceScale);
        
        // set shade to 1 if it's inside, 0 if it's outside        
    	shade = float(abs(uv.x) < 1.0 && abs(uv.y) < 1.0);
        
        // if it is behind the light source, don't light it!
        shade *= step(0.0, localPos.y);
        
        // apply distance attenuation
        shade *=  1.0 - clamp(directionalLightFalloff * localPos.y * localPos.y, 0.0, 1.0);        
    }
    
    // soften shadows over a distance
	if (GetMode() >= 2.0)
    {
        float softenDistance = clamp(localPos.y * directionalLightSoften, 0.01, 0.99);
    	float softenX = smoothstep(1.0, 1.0 - softenDistance, abs(uv.x));
    	float softenY = smoothstep(1.0, 1.0 - softenDistance, abs(uv.y));
    	shade = shade * softenX * softenY;
    }
    
    // apply texture to light if we should!
    if (GetMode() >= 3.0)
    {
        uv = uv*0.5+0.5;
        return clamp((texture(iChannel0, uv).rgb * directionalLightTextureMADD.x + directionalLightTextureMADD.y) * shade, 0.0, 1.0);
    }
    
    return vec3(shade);
}

// Function 73
vec3 Lighting(vec3 rd, vec2 uv, vec3 pos, vec3 norm, float ao, float shadow)
{
    vec3 sunLight = SunLightCol * clamp(dot(norm, SunLightDir), 0.0, 1.0) * 4.0;
    vec3 ambLight = AmbLightCol;
    
    vec3 reflVec   = reflect(-SunLightDir, norm);
    vec3 specLight = pow(max(0.0, dot(rd, -reflVec)), 2.0) * vec3(1.0) * shadow * 2.0;
    
    return (sunLight * shadow) + (ambLight * ao) + specLight;
}

// Function 74
void update_light_position(void)
{
    float sn = sin(iTime * 3.0);
    float cs = cos(iTime * 3.0);
    light_position = vec3(sn * 0.2,
                          0.9 + cs * cs * 0.2,
                          cs * 0.1);
}

// Function 75
vec3 Lighting_Sphere(
    in vec3  toView,
    in vec3  pos,
    in vec3  norm,
    in vec3  albedo,
    in float intensity,
    in float roughness)
{
    Light_Sphere light;
    light.position    = LightPos;
    light.attenuation = LightAttenuation;
    light.radius      = LightRadius;
    
	vec3  toLight     = normalize(light.position - pos);
    float lightCos    = max(0.0, dot(norm, toLight));
    float lightDist   = max(0.0, length(pos - light.position) - light.radius);  // Account for radius
    float attenuation = Attenuation(lightDist, light.attenuation.x, light.attenuation.y, light.attenuation.z);
    
    vec3 reflVector = reflect(-toView, norm);
    
    vec3 dirLight = vec3(intensity) * lightCos * attenuation;
    vec3 spcLight = vec3(0.2) * pow(max(0.0, dot(reflVector, toLight)), roughness) * lightCos;
    
    return (albedo * dirLight) + spcLight;
}

// Function 76
vec3 LightLine(vec3 p0, vec3 p1, vec3 p, vec2 uv, float N, float R, float I, int Type) {
    //N lights with size R (pixels) and intensity I on a line from p0 to p1
    vec3 Int=vec3(0.); vec2 Intensity;
    vec2 uv0=(p0.xy-p.xy)/(((p0.z-p.z)*CFOV)*ASPECT)*0.5+0.5;
    vec2 uv1=(p1.xy-p.xy)/(((p1.z-p.z)*CFOV)*ASPECT)*0.5+0.5;
    vec2 uv01=uv1-uv0;
    float uvlensqr=dot(uv01,uv01);
    if (uvlensqr==0.) {
        //Projection: point
    } else {
        //Projection: line
        float k=dot(uv-uv0,uv01)/uvlensqr;
        k=clamp(floor(k*N),0.,N-2.)/(N-1.);
        vec3 LightPos=p0+k*(p1-p0);
        Intensity=((length((uv0+uv01*k-uv)*ASPECT)>R*IRES.y)?vec2(0.,10000.):vec2(I,LightPos.z-p.z));
        k+=1./(N-1.);
        LightPos=p0+k*(p1-p0);
        Intensity=((length((uv0+uv01*k-uv)*ASPECT)>R*IRES.y)?Intensity:vec2(I,LightPos.z-p.z));
        Int[Type]=Intensity.x; Int.z=Intensity.y;
    }
    return Int;
}

// Function 77
vec3 GetSceneLight(float specLevel, vec3 normal, RayHit rayHit, vec3 rayDir, vec3 origin)
{        
  vec3 reflectDir = reflect( rayDir, normal );

  float amb = clamp( 0.5+0.5*normal.y, 0.0, 1.0 );
  float dif = clamp( dot( normal, sunPos ), 0.0, 1.0 );
  float bac = clamp( dot( normal, normalize(vec3(-sunPos.x, 0.0, -sunPos.z))), 0.0, 1.0 )*clamp( 1.0-rayHit.hitPos.y, 0.0, 1.0);
  float fre = pow( clamp(1.0+dot(normal, rayDir), 0.0, 1.0), 2.0 );
  specLevel*= pow(clamp( dot( reflectDir, sunPos ), 0.0, 1.0 ), 16.0);

  float skylight = smoothstep( -0.1, 0.1, reflectDir.y );
  vec3 shadowPos = origin+((rayDir*rayHit.depth)*0.98);  
  dif *= SoftShadow( shadowPos, sunPos);
  skylight *=SoftShadow(shadowPos, reflectDir);

  vec3 lightTot = vec3(0.0);

    
    
  lightTot += 1.30*dif*vec3(1.00, 0.80, 0.55);
  lightTot += 0.50*skylight*vec3(0.40, 0.60, 1.00);
      lightTot += 1.20*specLevel*vec3(0.9, 0.8, 0.7)*dif;
  lightTot += 0.50*bac*vec3(0.25, 0.25, 0.25);
  lightTot += 0.25*fre*vec3(1.00, 1.00, 1.00);
  return lightTot +(0.40*amb*vec3(0.40, 0.60, 1.00));
}

// Function 78
vec4 GetDirectionalLightRotationAxisAngle ()
{
    // mode >= 5.0 starts moving and rotating light source
    float time = GetMode() - 5.0;
    time = max(time, 0.0);
    
    vec4 ret = directionalLightRotationAxisAngle;
    ret.xyz += vec3(sin(time * 0.1), sin(time * 0.7), sin(time * 0.3));
    ret.xyz = normalize(ret.xyz);
    ret.w += time * 0.66;
        
    return ret;
}

// Function 79
vec3 lighting(vec3 p, vec3 n) {
    //AMBIENT (currently 0) Should be multiplied by diffuse color.
    vec3 retval = vec3(0., 0.2, 0.3); 
    float dprod = dot(n, ortho_light_dir);
    if(dprod >= 0.0) {
        retval += ortho_light_color * (
            //DIFFUSE
            vec3(0.85, 0.9, 1.0) * dprod +
            //SPECULAR
            //pow(dprod, 40.0)
            0.0
        );
    }
    return retval;
}

// Function 80
mat4 lightToWorldMatrix()
{
    return viewMatrix(lightPosition(), LIGHT_CAMERA_TARGET, UP);
}

// Function 81
vec3 sampleLight(vec3 v)
{
    return texture (iChannel1, v).rgb;
}

// Function 82
vec3 getLight(vec3 p, vec3 n, vec3 col)
{
    if (mat == NONE) return col;
    
    vec3 cr = cross(camDir, vec3(0,1,0));
    vec3 up = normalize(cross(cr,camDir));
    vec3 lp = vec3(-0.5,1.0,4.0); 
    vec3 l = normalize(lp-p);
 
    float ambient = 0.005;
    float dif = clamp(dot(n,l), 0.0, 1.0);
    
    if (mat == PUPL)
    {
        dif = clamp(dot(n,normalize(mix(camPos,lp,0.1)-p)), 0.0, 1.0);
        dif = mix(pow(dif, 16.0), dif, 0.2);
        dif += 1.0 - smoothstep(0.0, 0.2, dif);
        if (mat == PUPL) ambient = 0.1;
    }
    else if (mat == BULB)
    {
        dif = mix(pow(dif, 32.0), 3.0*dif+1.0, 0.2);
        ambient = 0.12;
    }
    else if (mat == PLANE)
    {
        dif = mix(pow(dif, 2.0), dif, 0.2);
    }
    
    if (mat == PLANE || mat == BULB)
    {
        dif *= softShadow(p, lp, 6.0);        
    }
       
    col *= clamp(dif, ambient, 1.0);
    col *= getOcclusion(p, n);
    
    if (light) col = vec3(dif*getOcclusion(p, n));
    
   	if (mat == PUPL || mat == BULB)
    {
        col += vec3(pow(clamp01(smoothstep(0.9,1.0,dot(n, l))), 20.0));
    }
    else if (mat == PLANE)
    {
        col += col*vec3(pow(clamp01(smoothstep(0.25,1.0,dot(n, l))), 2.0));
        col += col*vec3(pow(clamp01(smoothstep(0.9,1.0,dot(n, l))), 4.0));
    }
    
    if (light) col = clamp(col, 0.0, 1.0);
    return col;
}

// Function 83
vec3 lighting(const Ray ray, const RaycastHit hit) {
 
    float diffuse = ambientLight;
    vec3 l1 = light1 - hit.point;
    
    const int n = 30;
    int h = 0;
    
    for(int s = 0; s < n; s++) {
    
        vec3 s1 = (light1 + noise(vec4(hit.point.xyz,iTime + float(s)))) - hit.point;
    
    	Ray shadow;
    	shadow.origin = hit.point + (s1 * 0.0001);
    	shadow.direction = s1;
    
    	RaycastHit scene = raycastScene(shadow);
        
    	if(scene.type < 0) {
     
        	h++;
      
    	}
        
    }
        
    diffuse += max(dot(normalize(hit.normal),normalize(l1)),0.) * float(h) / float(n);
    return vec3(diffuse);
    //return vec3(scene.type + 1);
    
}

// Function 84
vec3 lightPath(in Ray ray, in float step_, in float counter)
{
    return ray.P + (step_ * counter * ray.V);
}

// Function 85
vec3 light(vec3 world, vec3 wsn)
{
    vec3 col = vec3(0.0);
    float lm = 1.0;
    vec3 lps = lightpos(world);
    vec3 lp = world + lps;
    vec3 ublv = world - lp;
    vec3 lv = ublv + wsn * 0.01;
    float ld = length(lv);
    lv /= ld;
    float lt = trace(lp, lv);
    if (lt >= ld) {
		vec3 plane = vec3(1.0, 0.0, 0.0);
        vec3 porg = vec3(3.5, 0.0, 0.0);
        vec3 del = porg - world;
        float x = dot(del, plane) / dot(normalize(ublv), plane);
		vec3 proj = world + lv * x;
        col = glassrep(proj);
    }
    return col;
}

// Function 86
vec3 SampleLight(vec2 uv) {
    //Returns mapped intensity in point uv
    vec2 tmp_v=Read2(texture(iChannel1,uv).z);
    return ITC(tmp_v.x)+ITC(tmp_v.y);
}

// Function 87
vec3 getLighting(vec3 color, float nDotL, vec3 lPos, vec3 rV){
    
    float reflection = pow(clamp(dot(lPos, rV), 0.0, 1.0), 10.0);
    
    return color * (sunlight * nDotL + (ambientLight + sunlight * 0.015)
           + (doSpecular ? reflection * sunlight : vec3(0.0)));
}

// Function 88
vec3 calcLighting(vec3 col, vec3 p, vec3 n, vec3 r, float sh, float sp) {
    float d = max(dot(LIGHT_DIR,n),0.);
    float s = 0.;
    float sd = 1.;
    if(raymarch(p+LIGHT_DIR*SHADOW_BIAS,LIGHT_DIR,32) < MAX_DISTANCE)
        sd = 0.;
    if(sh > 0.)
        s = pow(max(dot(LIGHT_DIR,r),0.),sh)*sp;
    d *= sd;
    s *= sd;
    return (col*(LIGHT_AMB+LIGHT_COL*d))+(LIGHT_COL*s);
}

// Function 89
void initLights() {
    #if (NUM_LIGHTS != 0)

    lights[0] = light(1.,//1 = point, 0 = direction
                      10.0,//importance
                      vec3(0.));//point pos or direction


    #endif
}

// Function 90
vec3 directLight(vec3 pos,vec3 normal
){//return vec3(0.)
 ;float dotLight=-dot(normal,LightDir)
 ;if(dotLight<0.0)return vec3(0)
 ;vec3 pos0=pos
 ;float minAngle=LightRadius
 ;for(int i=0;i < MaxShadowSteps;i++
 ){float dist=sdf(pos)
  ;if(dist>MaxDist)break
  ;if(dist<MinDist)return vec3(0)
  ;pos-=LightDir*dist*2.5//goes 2.5 times faster since we don't need details
  ;minAngle=min(asin(dist/length(pos-pos0)),minAngle);}
 ;return LightColor*dotLight*sat(minAngle/LightRadius);}

// Function 91
vec3 GetDirectionalLightSourcePosition ()
{
    // mode >= 5.0 starts moving and rotating light source
    float time = GetMode() - 5.0;
    time = max(time, 0.0);
    float canMove = step(5.0, GetMode());

    vec3 ret = directionalLightSourcePosition;
    
    ret += vec3(sin(time * 0.83) + 1.0 * canMove, sin(time * 1.1), sin(time * 0.1));
    
    return ret;
}

// Function 92
vec3 DirectLight(vec3 P, vec3 N, vec3 LD) {
    float SunDot=dot(LD,N); Hit Shad;
    return ((SunDot<0.)?vec3(0.):SunColor*(SunDot*(1.-float(SRay(P+N*0.01,LD,Shad)))));
}

// Function 93
vec3 light(ray r,vec3 sunPos,vec3 orig){
    if(distance(r.ro,center)<atm.r){
    }else{
        float d1=sphere(r,atm,1.);
        if(d1==0.)return orig;
        r.ro+=r.rd*d1;
    }
    float d2=sphere(r,earth,1.);
    if(d2==0.)d2=sphere(r,atm,-1.);
    float viewDepth=0.;
    vec3 l=vec3(0.);
    for(int i=0;i<VIEW_SAMPLES;i++){
        vec3 p=r.ro+r.rd*(float(i)+0.5)/float(VIEW_SAMPLES+1)*d2;
        ray k=ray(p,normalize(sunPos-p));
        #ifdef EXPERIMENTAL_PLANET_SHADOW
        if(sphere(k,earth,1.)==0.){
        #endif
            float sunDepth=depth(k.ro,k.ro+k.rd*sphere(k,atm,-1.));
            viewDepth=depth(r.ro,p);
            vec3 transmitance=exp(-(sunDepth+viewDepth)*RGBScatter);
        
        
            l+=transmitance*density(p)*phase(dot(r.rd,normalize(sunPos-p)));
        #ifdef EXPERIMENTAL_PLANET_SHADOW
        }
        #endif
    }
    vec3 origTransmitance=exp(-viewDepth*RGBScatter);
    return orig*origTransmitance+l/float(VIEW_SAMPLES)*d2*sunInt*RGBScatter*scatterStrength;
}

// Function 94
float lightRay(vec3 org, vec3 p, float phaseFunction, float mu, vec3 sunDirection){
    float lightRayDistance = CLOUD_EXTENT*0.75;
    float distToStart = 0.0;

    getCloudIntersection(p, sunDirection, distToStart, lightRayDistance);

    float stepL = lightRayDistance/float(STEPS_LIGHT);

    float lightRayDensity = 0.0;

    float cloudHeight = 0.0;

    //Collect total density along light ray.
    for(int j = 0; j < STEPS_LIGHT; j++){
        //Reduce density of clouds when looking towards the sun for more luminous clouds.
        lightRayDensity += mix(1.0, 0.75, mu) * 
            clouds(p + sunDirection * float(j) * stepL, cloudHeight);
    }

    //Multiple scattering approximation from Nubis presentation credited to Wrenninge et al. 
    //Introduce another weaker Beer-Lambert function.
    float beersLaw = max(exp(-stepL * lightRayDensity), 
                         exp(-stepL * lightRayDensity * 0.2) * 0.75);

    //Return product of Beer's law and powder effect depending on the 
    //view direction angle with the light direction.
    return mix(beersLaw * 2.0 * (1.0-(exp(-stepL*lightRayDensity*2.0))), beersLaw, mu);
}

// Function 95
vec4 doLighting(vec3 eyePoint, vec3 objPoint, vec3 normalAtPoint, vec3 lightPos, vec4 lightParams) {
	float fresnelBias = lightParams.x;
	float fresnelPower = lightParams.y;
	float fresnelScale = lightParams.z;
	float constAttenuation = 9000000.0;
	float linearAttenuation = 0.22;
	float quadraticAttenuation = 0.2;
	float dist = length(lightPos-objPoint);
	float attenuation = constAttenuation / ((1.0+linearAttenuation*dist)*(1.0+quadraticAttenuation*dist*dist));
	float shininess = lightParams.w;
	vec3 I = normalize(objPoint - eyePoint);
	vec3 lightDirection = normalize(lightPos-objPoint);
	vec3 viewDirection = normalize(eyePoint-objPoint);
	vec3 halfVector = normalize(lightDirection + viewDirection);
	float dif = clamp(dot(normalAtPoint, lightDirection), 0.0, 1.0);
	float spec = max(0.0, pow(dot(normalAtPoint, halfVector), shininess));
	float fresnel = clamp(fresnelBias + fresnelScale * pow(1.0 + dot(I, normalAtPoint), fresnelPower), 0.0, 1.0);
	return attenuation * vec4(vec3(mix(spec, dif, fresnel)), 1.0);
}

// Function 96
void initLightSphere( float time ) {
	lightSphere = vec4( 3.0+2.*sin(time),2.8+2.*sin(time*0.9),3.0+4.*cos(time*0.7), .5 );
}

// Function 97
vec3 Lighting_Point(
    in vec3  toView,
    in vec3  pos,
    in vec3  norm,
    in vec3  albedo,
    in float intensity,
    in float roughness)
{
    Light_Point light;
    light.position    = LightPos;
    light.attenuation = LightAttenuation;
    
	vec3  toLight     = normalize(light.position - pos);
    float lightCos    = max(0.0, dot(norm, toLight));
    float lightDist   = length(pos - light.position);
    float attenuation = Attenuation(lightDist, light.attenuation.x, light.attenuation.y, light.attenuation.z);
    
    vec3 reflVector = reflect(-toView, norm);
    
    vec3 dirLight = vec3(intensity) * lightCos * attenuation;
    vec3 spcLight = vec3(0.2) * pow(max(0.0, dot(reflVector, toLight)), roughness) * lightCos;
    
    return (albedo * dirLight) + spcLight;
}

// Function 98
vec3 DirectLightSample(in Intersection intersecNow,out vec3 wi,out float pdf){
	vec3 Li = vec3(0.);
    float x1 = GetRandom(),x2 = GetRandom();
    float dist = INFINITY;
    vec3 AssumeLi = LightSample(intersecNow.surface,x1,x2,wi,dist,pdf);
    Ray shadowRay = Ray(intersecNow.surface,wi);
    Intersection intersecNext;
    SceneIntersect(shadowRay, intersecNext);
    if(intersecNext.type == LIGHT){
    	Li = AssumeLi;
    }
    return Li;
}

// Function 99
vec3 LightRing(vec3 cp, float r, vec3 p, vec3 dir, float N, float R, float I, int Type) {
    //N lights with size R (pixels) and intensity I on a circle with normal e_y, radius r and center p
    vec3 Int=vec3(0.); vec2 Intensity;
    if (dir.y==0.) {
        //Projection: line
    } else {
        //Projection: ellips
        vec2 CircleCoord=(p-cp+dir*(-(p.y-cp.y)/dir.y)).xz;
        float Angle=atan(CircleCoord.x,CircleCoord.y)+RADIAN*0.5;
        float FAngle=floor(Angle*IRADIAN*N)*(RADIAN/N);
        vec3 LightPos=cp+vec3(-r*sin(FAngle),0.,-r*cos(FAngle));
        Intensity=((TraceSphere(p,dir,LightPos,R*IRES.y*(LightPos.z-p.z))<0.)?
                                    vec2(0.,10000.):vec2(I,LightPos.z-p.z));
        FAngle+=(RADIAN/N);
        LightPos=cp+vec3(-r*sin(FAngle),0.,-r*cos(FAngle));
        Intensity=((TraceSphere(p,dir,LightPos,R*IRES.y*(LightPos.z-p.z))<0.)?
                                    Intensity:vec2(I,LightPos.z-p.z));
        Int[Type]=Intensity.x; Int.z=Intensity.y;
    }
    return Int;
}

// Function 100
vec3 lighting(vec3 sp, vec3 sn, vec3 lp, vec3 rd)
{
vec3 color;
    
    //some other experiemnts
    //where the id's are based on cells you don't need to pass the id variable around
    //you can just recreate it where needed.
    /*float id = rnd(floor(sp.xz));
    float id1to3 = floor(id*3.0);
    float one = step(1., id1to3);
    float two = step(2., id1to3);
    float three = step(3., id1to3);///hmmm*/
    
    //vec3 tex = texture(iChannel0, sp.xz).xyz*one;
    vec3 lv = lp - sp;
    float ldist = max(length(lv), 0.001);
    vec3 ldir = lv/ldist;
    
    float atte = 1.0/(1.0 + 0.002*ldist*ldist );
    
    float diff = dot(ldir, sn);
    float spec = pow(max(dot(reflect(-ldir, sn), -rd), 0.0), 10.);
    float fres = pow(max(dot(rd, sn) + 1., 0.0), 1.);
	float ao = calculateAO(sp, sn);
    
    vec3 refl = reflect(rd, sn);
    vec3 refr = refract(rd, sn, 0.7);
    
    
    vec3 str = stripes(sp);
    vec3 chessFail = vec3(floor(mod(sp.z, 2.))+floor(mod(sp.x, 2.)));
    float rndTile = rnd(floor(sp.xz*SCALE ));//+ iTime/10.);
    
    
    //color options
    vec3 color1 = vec3(rndTile*2., rndTile*rndTile, 0.1);
    vec3 color2 =vec3(rndTile*rndTile, .0, rndTile/90.);
    vec3 color3 =mix(vec3(0.9, 0., 0.), vec3(1.4), 1.0-floor(rndTile*2.));
    
    //getting reflected and refracted color froma cubemap, only refl is used
    vec4 reflColor = texture(iChannel1, refl);
    vec4 refrColor = texture(iChannel2, refr);
     
    //blue vs orage specular, orange all the way.
    vec3 coolSpec = vec3(.3, 0.5, 0.9);
    vec3 hotSpec = vec3(0.9,0.5, 0.2);
   
    
    //apply color options and add refl/refr options
    color = (diff*color2 +  spec*hotSpec +reflColor.xyz*0.2 )*atte;
	
    
    //apply ambient occlusion and return.
 return color*ao;   
}

// Function 101
float light_map(vec3 p, float time)
{
    float sd = sdSphere(p - vec3(-10.*cos(0.0*time),-10.*cos(0.0*time), 5.), 1.);
    //sd = min(sd, sdSphere(p - vec3(5.*cos(0.8*iTime+3.14),5.*sin(0.8*iTime+3.14), 2.), 0.5));
    return sd;
}

// Function 102
vec2 maplight(vec3 orp)
{
    float t = iTime * 0.025;
    float minm = 10000.0;
    float mm = 10000.0;
    float hit_ids = 0.0;
    
    for (int i = 0; i < SPOTS; ++i)
    {
	    vec3 rp = orp;
	    vec3 _rp = rp;
        rp += SPOT_POS[i];
        rp *= SPOT_ROTATION[i];
        
        float m = sdCappedCylinder(rp, vec2(CONE_W, 1.0));

        float l = -LIGHT_BASE_W + length(rp) * 0.2;
        m -= l;
        float d = dot(rp, vec3(0.0, -1.0, 0.0));
        
        if( m < 0.0 && d >= 0.0)
        {
            vec3 uv = _rp + vec3(t, 0.0, 0.0);
            float n = noise(uv * 10.0) - 0.5;
            
            uv = _rp + vec3(t * 1.2, 0.0, 0.0);
            n += noise(uv * 22.50) * 0.5;

            uv = _rp + vec3(t * 2.0, 0.0, 0.0);
            n += noise(uv * 52.50) * 0.5;
            
            uv = _rp + vec3(t * 2.8, 0.0, 0.0);
            n += noise(uv * 152.50) * 0.25;

            mm = min(n, m);
            mm = min(mm, -0.2);
            hit_ids += float(i + 1);

        }
        minm = min(abs(m), minm);
    }
    
    if(hit_ids > 0.0)
    {
        return vec2(mm, hit_ids);
    }
    
    return vec2(minm, NOTHING);

}

// Function 103
vec3 GetSceneLight(float specLevel, vec3 normal, vec3 pos, vec3 rayDir)
{      
  vec3 light1 = normalize(vec3(-1.0, 2.8, 1.0));
    
  vec3 reflectDir = reflect( rayDir, normal );
  specLevel *= pow(clamp( dot( reflectDir, light1 ), 0.0, 1.0 ), 16.0);

  float amb = clamp( 0.5+0.5*normal.y, 0.0, 1.0 );
  float diffuse = clamp( dot( normal, light1 ), 0.0, 1.0 );
  float skyLight = smoothstep( -0.1, 0.1, reflectDir.y );
  float fill = pow( clamp(1.0+dot(normal, rayDir), 0.0, 1.0), 1.0 )*1.0;
  float backLight = clamp( dot( normal, normalize(vec3(-light1.x, 0.0, -light1.z))), 0.0, 1.0 )*5.0;

  diffuse *= SoftShadow( pos, light1);
  skyLight *= SoftShadow( pos, reflectDir);

  vec3 lightTot = 1.30*diffuse*vec3(1.00, 0.80, 0.55);
  lightTot += specLevel*vec3(1.00, 0.90, 0.70)*diffuse;
  lightTot += 0.40*amb*vec3(0.40, 0.60, 1.00);
  lightTot += 0.50*skyLight*vec3(0.40, 0.60, 1.00);
  lightTot += 0.50*backLight*vec3(0.25, 0.25, 0.25);

  return lightTot+(0.25*fill*vec3(1.00, 1.00, 1.00));
}

// Function 104
float lightPointDiffuseSoftShadow(vec3 pos, vec3 lightPos, vec3 normal) {
	vec3 lightDir = normalize(lightPos - pos);
	float lightDist = length(lightPos - pos);
	float color = max(dot(normal, lightDir), 0.0) / (lightDist * lightDist);
	if (color > 0.00) color *= castSoftShadowRay(pos, lightPos);
	return max(0.0, color);
}

// Function 105
void vs_pace_flight_dynamics( inout VehicleState vs, LocalEnvironment env, float dt, inout vec3 dv, inout vec3 dL )
{
    vec3 uvw = 1000. * vs.localv * vs.localB;
    vec3 pqr = vs.localomega * vs.localB;
    vec3 uvwdot = ZERO, pqrdot = ZERO;
    float invtau = 2. * abs( uvw.x ) / ( g_vehicle_data[ USE_VEHICLE_INDEX ].Sbcm.z );

    compute_flight_dynamics(
        g_vehicle_data[ USE_VEHICLE_INDEX ],
        env.atm.z,
        env.atm.w * 1000.,
        uvw,
        pqr,
        vec3( -vs.EAR.x, vs.EAR.yz ),
        vs.FSG,
        vs.wdelay,
        uvwdot,
        pqrdot,
        vs.info.xyz );

    vs.wdelay -= expm1( -dt * invtau ) * ( uvw.z - vs.wdelay );
    dv += dt * vs.localB * ( uvwdot /* - cross( uvw, pqr ) */ ) / 1000.;
    dL += dt * vs.localB * pqrdot;
}

// Function 106
Intersection intersectLight(in Ray ray, in Light light)
{
    Quadric quadric;
    quadric.p = light.p;
    quadric.r = vec3(light.r);
    
    return intersectQuadric(ray, quadric);
}

// Function 107
vec3 computeSaberLightDir(vec3 point, StickmanData data, FieldData fieldData)
{
#if LIGHT_QUALITY
    float grad = 0.75;
    
    vec3 normalWS = vec3(0.0);
    for(int i = min(0, iFrame); i < 4; i++)
    {
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        uint unusedMatId;
        vec3 p = point + e * grad;
        normalWS += e*fSaber(data.invStickmanRot*(p - data.stickmanPos), data, fieldData);
    }
    normalWS = -normalize(normalWS); 
    return normalWS;
#else
    vec3 s = (vec3(0., data.saberLen*2.0, 0.)*data.invSaberRot);
    vec3 a = data.stickmanPos + (data.bodyPos + data.saberPos*data.bodyDeform)*data.invStickmanRot;
    vec3 c = data.stickmanPos + (data.bodyPos + (s + data.saberPos)*data.bodyDeform)*data.invStickmanRot;
    return normalize(mix(a, c, 0.5) - point);
#endif
}

// Function 108
vec4 light( in vec3 pos )
{
    vec4 e = vec4(0.0005,-0.0005, 0.25, -0.25);
    return   (e.zwwz*light_map( pos + e.xyy , iTime) + 
  			  e.wwzz*light_map( pos + e.yyx , iTime) + 
			  e.wzwz*light_map( pos + e.yxy , iTime) + 
              e.zzzz*light_map( pos + e.xxx , iTime) )/vec4(e.xxx, 1.);
}

// Function 109
vec3 Lighting_Circle(
    in vec3  toView,
    in vec3  pos,
    in vec3  norm,
    in vec3  albedo,
    in float intensity,
    in float roughness)
{
    Light_Ring light;
    light.position    = LightPos;
    light.attenuation = LightAttenuation;
    light.normal      = Rotate(LightNormal, LightYaw, LightPitch);
    light.radius      = LightRadius;
    
    // Get the closest point on or in the circle on the plane defined by
    // the light's position and normal. This point is then used
    // for the standard lighting calculations.
    
    vec3 pointOnLight = PointInCircle(pos, light.position, light.normal, light.radius);
    vec3 toLight      = pointOnLight - pos;
    vec3 toLightN     = normalize(toLight);
    
    float lightCos    = dot(-toLightN, light.normal);
          lightCos    = (LightDirection == LightDirectionBi) ? abs(lightCos) : max(0.0, lightCos);
    
    float lightDist   = length(toLight);
    float attenuation = Attenuation(lightDist, light.attenuation.x, light.attenuation.y, light.attenuation.z);
    
    vec3 reflVector = reflect(-toView, norm);
    
    vec3 dirLight = vec3(intensity) * lightCos * attenuation;
    vec3 spcLight = vec3(0.2) * pow(max(0.0, dot(reflVector, toLightN)), roughness) * lightCos;
    
    return (albedo * dirLight) + spcLight;
}

// Function 110
void sampleDirectLight( vec3 pos,
                       	vec3 normal,
                        float Xi1,
                        float Xi2, 
                       	out vec3 dir,
                       	out float pdf ) {
    float height = objects[0].params_[2] - objects[0].params_[1];
    float r = objects[0].params_[0];
    float pdfA;
    float d2;
    float aCosThere;
    float theta;
    float thetaPdf;
    float h;
    float hPdf;
    
    //convert position to object space
    pos = toVec3( objects[0].transform_inv_*vec4(pos, 1.0) );
    normal = toVec3( objects[0].transform_inv_*vec4(normal, 0.0) );
    
    if(samplingTechnique == SAMPLE_TOTAL_AREA){
        theta = Xi1*TWO_PI;
        thetaPdf = 1.0/TWO_PI;
        h = objects[0].params_[1] + Xi2*height;
        hPdf = 1.0/height;
        
        vec3 n = vec3(cos(theta), sin(theta), 0.0);
        vec3 p = n*r;
        p.z = h;
        dir = p - pos;
        d2 = dot(dir,dir);
        dir /= sqrt(d2);
        aCosThere = max(0.0,dot(-dir,n));

        pdfA = thetaPdf*hPdf*(1.0/r);
        pdf = PdfAtoW( pdfA, d2, aCosThere );
    } else {
        vec3 cylinderPos = vec3(0.0, 0.0, objects[0].params_[1]);
        vec3 cylinderVec = vec3(0.0, 0.0, 1.0);
        
        float dc_2 = dot(pos.xy,pos.xy);
        float dc = sqrt(dc_2);
        vec2 dirToPos = (-pos.xy)/dc;
        float alphaMax = acos(r/dc);
        float thetaMax = 0.5*PI - alphaMax;
        theta = mix(0.0,thetaMax, Xi1);
        float sinTheta = sin(theta);
        float sin2Theta = sinTheta*sinTheta;
        float cosTheta = sqrt(1.0-sin2Theta);
        
    	float ds = dc * cosTheta - sqrt(max(0.0, r*r - dc_2 * sin2Theta));
    	float cosAlpha = (dc * dc + r*r - ds*ds) / (2.0 * dc * r);
        
        float alpha = acos(cosAlpha);
        
        //convert alpha to global angle
        float baseAngle = atan(dirToPos.y,dirToPos.x) + PI;
        float sign;
        if(Xi2<0.5) {
            sign =  1.0;
            //normalize Xi
            Xi2 /= 0.5;
        } else {
            sign = -1.0;
            //normalize Xi
            Xi2 = (Xi2 - 0.5) / 0.5;
        }
        
        float alphaGlobal = baseAngle + alpha*sign;
       
        thetaPdf = 1.0/(2.0*thetaMax);// In angle measure
        thetaPdf = PdfWtoA(thetaPdf, ds, cos(alpha+theta));
        
        vec3 n = vec3(cos(alphaGlobal), sin(alphaGlobal), 0.0);
    	vec3 p = n*r;
        
        //Sampling h
        //We do equiangular sampling from shading point
        {
      		sampleLine( pos, normalize(normal), p, normalize(cylinderVec), height, Xi2, h, hPdf );
        }
        
        p.z = cylinderPos.z + h;
        dir = p - pos;
        d2 = dot(dir,dir);
        dir /= sqrt(d2);
        aCosThere = max(0.0,dot(-dir,n));

        pdfA = thetaPdf*hPdf;
        pdf = PdfAtoW( pdfA, d2, aCosThere );
    }
    
    //convert dir to world space
    dir = toVec3( objects[0].transform_*vec4(dir,0.0) );
}

// Function 111
vec3 lighting(in Hit h) {
  if (h.ray.len > MAX_TRACE_DIST) return textureLod(iChannel0, -h.ray.dir, 0.).rgb;
  vec4 fgi = gi(h.pos, h.nml);    // Fake Global Illumination
  vec4 fcs = cs(h.pos, dif.dir);  // Fake Caustic Shadow
  //   lin = ([Ambient]        + [Diffuse]        * [SS]  + [CAUSTICS])  * [AO]  + [GI]
  vec3 lin = (_lit(h.nml, amb) + _lit(h.nml, dif) * fcs.w + fcs.rgb) * fgi.w + fgi.rgb;
  return  h.srf.kd * lin;
}

// Function 112
float getLightFactor(in vec4 lightPos, in vec4 worldPos, in vec4 worldNormal, in Scene scene){
    vec4 diff = lightPos - worldPos;
    float lightDist = length(diff);
    vec4 lightDir = normalize(diff);
    float lightDot = clamp(dot(lightDir, worldNormal), 0.0, 1.0);
    
    //shadow
    Ray lightRay = Ray(scene.sunPos, -lightDir);
    RayHit lightHit = rayCastBase(lightRay, scene);    
    
    const float shadowBias = 0.01;
    
    return lightDot * float((lightHit.dist + shadowBias) > lightDist);
}

// Function 113
float getLightIntensity( const vec3 pos, const vec3 normal, const vec3 light, const float intensity) {
    vec3 rd = pos - light;
    float i = max(0., dot(normal, -normalize(rd)) / dot(rd,rd));
    i = i > 0.0001 ? i * intensity * shadowhit(light, normalize(rd), length(rd)) : 0.;
    return max(0., i-0.0001);              
}

// Function 114
float lightPhotonStartTime( vec3 finalPos, float finalTime )
{
    float startTime = finalTime;
    
    // my old friend FPI
    for( int i = 0; i < 3; i++ )
    {
        startTime = finalTime - light_time_per_m() * length( light( startTime ) - finalPos );
    }
    
    return startTime;
}

// Function 115
vec3 worldPointToLightPoint(vec3 worldPoint, out float lightDist)
{
    vec3 lightDir = lightDirection();
    vec3 lightPos = lightPosition();
    
    // Plane SDF allows easy distance calculation to ortographic plane.
    // -lightPos = lightDir * length(lightPos)
    lightDist = planeSDF(worldPoint, -lightPos);

    mat4 worldToLight = inverse(lightToWorldMatrix());
    
    // Quick maths
    vec3 lightPoint = worldPoint + lightDist * lightDir;
	lightPoint = (worldToLight * vec4(lightPoint, 0.0)).xyz;
    lightPoint = (lightPoint + LIGHT_CAMERA_SIZE) / (2.0 * LIGHT_CAMERA_SIZE);
    
    return lightPoint;
}

// Function 116
vec3 sampleLightE( in vec3 hitOrigin, in vec3 hitNormal, in vec3 rayDir, in Material material  )
{
    vec3 e = vec3( 0 );
    vec3 s = vec3( 0 );

    Light light;
    light.id = 3.0;
    light.emission = LIGHT1_EM;

    vec3 l0 = LIGHT1_POS - hitOrigin;

    float cos_a_max = sqrt(1. - clamp(0.5 * 0.5 / dot(l0, l0), 0., 1.));
    float cosa = mix(cos_a_max, 1., random());
    vec3 l = jitter(l0, 2.*PI*random(), sqrt(1. - cosa*cosa), cosa);

#if (PATH == 1)
    vec3 lightHit = castRay( hitOrigin, l, 0.001, 100.0 );
    if ( lightHit.z == light.id )
#else
    s += softshadow( hitOrigin, normalize(l0) );
#endif
    {
        float omega = 2. * PI * (1. - cos_a_max);
        vec3 n = normalize(hitOrigin - LIGHT1_POS);
        e += ((light.emission * clamp(dot(l, n),0.,1.) * omega) / PI);
    }

    light.id = 4.0;

    l0 = vec3( -4, 1.5, 4 ) - hitOrigin;

    cos_a_max = sqrt(1. - clamp(0.5 * 0.5 / dot(l0, l0), 0., 1.));
    cosa = mix(cos_a_max, 1., random());
    l = jitter(l0, 2.*PI*random(), sqrt(1. - cosa*cosa), cosa);

#if (PATH == 1)
    lightHit = castRay( hitOrigin, l, 0.001, 100.0 );
    if ( lightHit.z == light.id )
#else
    s += softshadow( hitOrigin, normalize(l0) );
#endif
    {
        float omega = 2. * PI * (1. - cos_a_max);
        vec3 n = normalize(hitOrigin - vec3( -4, 1.5, 4 ));
        e += ((light.emission * clamp(dot(l, n),0.,1.) * omega) / PI);
    }

#if (PATH == 0)
    e *= clamp( s, 0., 1. );
#endif

    return e;
}

// Function 117
vec3 traceLight( in vec3 from, in vec3 norm, in uvec3 seed ) {
    vec3 pos = vec3(0);
    vec3 diff = vec3(0);
    vec3 dummyNorm = vec3(0);
    
    // create a random dir in a hemisphere
    vec3 rand = hash(seed);
    float dirTemp1 = 2.0*PI*rand.x;
    float dirTemp2 = sqrt(1.0-rand.y*rand.y);
    vec3 dir = vec3(
        cos(dirTemp1)*dirTemp2,
        sin(dirTemp1)*dirTemp2,
        rand.y);
    dir.z = abs(dir.z);
    
    // pick the sun more often (importance sampling)
    const float sunContrib = colorSun.a*2.0*PI*(1.0 - sunCosAngle);
    const float ambientContrib = colorAmbient.a*2.0*PI;
    const float sumContrib = sunContrib+ambientContrib;
    
    float a = sunContrib / sumContrib;
    float b = a + ambientContrib / sumContrib;
    
    if (rand.z < a) {
        const vec3 sunDirTan = normalize(cross(sunDir, vec3(0, 0, 1)));
        const vec3 sunDirCoTan = cross(sunDir, sunDirTan);
        float rot = 2.0*PI*rand.x;
        float the = acos(1.0 - rand.y*(1.0 - cos(sunAngle)));
        float sinThe = sin(the);
        dir = sunDirTan*sinThe*cos(rot) +
            sunDirCoTan*sinThe*sin(rot) - sunDir*cos(the);
    }
    
    if (trace(from, dir, false, pos, dummyNorm, diff)) {
        vec3 back = getBackground(dir);
        float l = max(0.0, dot(dir, norm));
        return back*l*sumContrib;
    } else {
        return vec3(0);
    }
}

// Function 118
vec4 PointLight(int val, in IntersectResult ir){
    vec3 a,b, nor;  
    vec3 v = lights[val].xyz-ir.p;
    float dist = castShadowRay( ir.p+ir.n*.02, normalize(v), a, b, nor );
    dist = dist < length(v) ? 0.4: 1.;
    return vec4(pow(max(1.-min(1.,length(lights[val].xyz-ir.p)/lights[val].w),0.),3.))*dist;
}

// Function 119
vec3 light(vec3 N) {
    vec3 irradiance = vec3(0.0);
        
	vec3 up    = vec3(0.0, 1.0, 0.0);
	vec3 right = cross(up, N);
	up         = cross(N, right);
 
	float nrSamples = 0.0; 
	for(float phi = 0.0; phi < 2.0 * PI; phi += sampleDelta)
	{
    	for(float theta = 0.0; theta < 0.5 * PI; theta += sampleDelta)
    	{
        	vec3 tangentSample = vec3(sin(theta) * cos(phi),  sin(theta) * sin(phi), cos(theta));
        	vec3 sampleVec = tangentSample.x * right + tangentSample.y * up + tangentSample.z * N; 
            
            irradiance += texture(iChannel0, sampleVec).xyz * 2.0 * cos(theta) * sin(theta);
       	 	nrSamples++;
    	}
	}
	irradiance = PI * irradiance * (1.0 / nrSamples);
    return irradiance;
    
}

// Function 120
vec3 lightSample( const in LightInfo light, const in SurfaceInteraction interaction, out vec3 wi, out float lightPdf, float seed, const in MaterialInfo material) {
    vec3 L = (light.position - interaction.point);
    vec3 V = -normalize(interaction.incomingRayDir);
    vec3 r = reflect(V, interaction.normal);
    vec3 centerToRay = dot( L, r ) * r - L;
    vec3 closestPoint = L + centerToRay * clamp( light.radius / length( centerToRay ), 0.0, 1.0 );
    wi = normalize(closestPoint);


    return light.L/dot(L, L);
}

// Function 121
vec3 GetLightPos()
{
    vec2 ndcMousePos = FromPixelsCoordToNDC(iMouse.xy);
    //return vec3(ndcMousePos, -2.);
    return vec3(2.0*ndcMousePos.x,0, 0);
}

// Function 122
vec3 GetLightDir(in vec3 position)
{
	//segment light
	float lambda = dot(position-lightPos,lightVec.xyz); 
	vec3 closestLightPos = clamp(lambda,0.0,lightVec.w)*lightVec.xyz+lightPos;
	
	return closestLightPos - position; 
}

// Function 123
vec3 doLighting(vec3 surfColor, vec3 surfPoint, vec3 surfNormal, vec3 lightColor, vec3 lightDir, float roughness)
{
    if(dot(surfColor, surfColor) <= 2.5)
    {             
        float lightingWrap = 0.5;
        float diff 		= dot(surfNormal, lightDir);
        diff			= clamp((diff + lightingWrap)/((1.0 + lightingWrap)*(1.0 + lightingWrap)), 0.0, 1.0);
        vec3 eyeDir 	= normalize(projectionCenter - surfPoint);
        vec3 halfVec 	= normalize(eyeDir + lightDir);
        float spec 		= clamp(dot(halfVec, surfNormal), 0.0, 1.0);
        spec 			= pow(spec, mix(128.0, 8.0, pow(roughness, 0.5))) * pow(diff, 1.0);
        surfColor 		= surfColor * diff * lightColor + lightColor * spec;
    }    
    return surfColor;
}

// Function 124
vec3 evaluatePointLight(vec3 pos, vec3 normal, vec3 lightDir, vec3 albedo)
{
    vec3 viewDir = normalize(cameraPos - pos);
 	vec3 halfVec = normalize(viewDir + lightDir);
    float nlDot = dot(normal, lightDir);
    if (nlDot <= 0.) return vec3(0.);
    float hlDot = dot(halfVec, lightDir);
    float nhDot = dot(normal, halfVec);
    float nvDot = dot(normal, viewDir);
    
    float fresnel = evaluateSchlickFresnel(nvDot);
    float refl = mix(0.03, 1., fresnel);
    float roughness = 0.2;
    float spec = evaluateGGXSpecularDistribution(nhDot, roughness)
        * evaluateBeckmannGeometryShadowing(nlDot, nvDot, roughness);
    return mix(albedo, vec3(spec), refl) * nlDot;
}

// Function 125
float GetLight_PDF(in Intersection intersecNow,in Intersection intersecNext){
	float pdf = 0.;
    if(intersecNext.type == LIGHT){
        vec3 v0v1 = quads[0].v1 - quads[0].v0;
    	vec3 v0v3 = quads[0].v3 - quads[0].v0;
    	float width  = length(v0v1);
    	float height = length(v0v3);
        vec3 lDir = intersecNext.surface - intersecNow.surface;
        float dist = length(lDir);
        float costhe = dot(-lDir,quads[0].normal);
        pdf = PDF_Area2Angle(1./(width*height),dist,costhe);
    }
    return pdf;
}

// Function 126
float sphereLight( vec3 pos, vec3 N, vec3 V, vec3 r, float f0, float roughness, float NoV, out float NoL )
{
	vec3 L				= spherePos - pos;
	vec3 centerToRay	= dot( L, r ) * r - L;
	vec3 closestPoint	= L + centerToRay * clamp( sphereRad / length( centerToRay ), 0.0, 1.0 );	
	vec3 l				= normalize( closestPoint );
	vec3 h				= normalize( V + l );
	
	NoL				= clamp( dot( N, l ), 0.0, 1.0 );
	float HoN		= clamp( dot( h, N ), 0.0, 1.0 );
	float HoV		= dot( h, V );
	
	float distL		= length( L );
	float alpha		= roughness * roughness;
	float alphaPrime	= clamp( sphereRad / ( distL * 2.0 ) + alpha, 0.0, 1.0 );
	
	float specD		= specTrowbridgeReitz( HoN, alpha, alphaPrime );
	float specF		= fresSchlickSmith( HoV, f0 );
	float specV		= visSchlickSmithMod( NoL, NoV, roughness );
	
	return specD * specF * specV * NoL;
}

// Function 127
float GetLight(vec3 p){
    //position of the light source
   // vec3 lightPos = vec3(40,100,0);
    
    //lightPos.xz += vec2(sin(iTime),cos(iTime));
    //light vector
    vec3 l = normalize(lightPos-p);
    
    //normal of object
    vec3 n = GetNormal(p);
    
    // dot product of the light vector and normal of the point
    // will give us the amount of lighting to apply to the point
    // dot() evaluates to values between -1 and 1, so we will clamp it
    float diff = clamp(dot(n, l),0.,1.);
    
    // calculate if point should be a shadow:
    // raymarch from point being calculated towards light source
    // if hits surface of something else before the light,
    // then it must be obstructed and thus is a shadow
    // the slight offset "p+n*SURFACE_DIST*1.1" is needed to ensure the
    // break condistions in the function are not met too early
    float d = RayMarch(p+n*SURFACE_DIST*1.1,l).x;
    if(d < length(lightPos-p)){
        diff *= .1;
    }
    
    return diff;
}

// Function 128
float getSphereLightIntensity(float num) {
    return num > .5 ?
        clamp(fract(time)*10.-1., 0., 1.) :
		max(0., 1.-fract(time)*10.); 
}

// Function 129
vec3 lightSurfacePoint(vec3 eye, vec3 surfacePoint, vec3 surfaceNormal, float ambientLight, int materialPick
){vec3 surfaceColour = vec3(0)
 ;float shadow = 1.
 ;for (int l = 0; l < NUM_LIGHTS; l ++
 ){vec3 lightPos =getLightPosition(l)//=vec3(15,40,35)
  ,L=normalize(lightPos-surfacePoint)                     
  ,colour=colours[materialPick]*.01
  ;float dotLN=dot(L,surfaceNormal)
  
  ;	if (dotLN < 0.0
  ){// Light not visible from this point on the surface
   ;colour =  colours[materialPick] * 0.01
  ;}else{
   ;float dotRV=dot(normalize(reflect(-L,surfaceNormal)),normalize(eye-surfacePoint))
      ;if (dotRV<0.){
        	colour = intensities[l] * (colours[materialPick] * dotLN);
            shadow = min(shadow, shadowMarch(surfacePoint, lightPos));
    	}else{
    		colour = intensities[l] * (colours[materialPick] * dotLN + speculars[materialPick] * pow(dotRV, shine[materialPick]));
            shadow = min(shadow, shadowMarch(surfacePoint,lightPos));
        }}
        shadow=mix(.2,1.,shadow)
       ;       surfaceColour += colour;}
	return surfaceColour*shadow*ambientLight;}

// Function 130
vec3 addLight(in vec3 lpos, inout vec3 col, in vec3 pos, in vec3 nor, in vec3 rd, in float thi){
	vec3 ldir = normalize(lpos-pos);
	float latt = pow( length(lpos-pos)*.03, .5 );
    float trans =  pow( clamp( max(0.,dot(-rd, -ldir+nor)), 0., 1.), 1.) + 1.;
	//col = vec3(.2,.1,.1) * (max(dot(nor,ldir),0.) ) / latt;
	col += vec3(.3,.3,.1) * (trans/latt)*thi;
    return col;
   
}

// Function 131
float GetLightmapSphereFaceRes(Object obj)
{
    // assume 3x2 grid ..
	return (obj.lightmapBounds.w - obj.lightmapBounds.y)*0.5;
}

// Function 132
float D_GGX_AreaLight( float NDotH, float Alpha, float AlphaPrim )
{
    float AlphaSqr = Alpha*Alpha;
    float GGXNormalizationFactor = (1. / AlphaSqr) * ONE_OVER_PI;
    float GGXNormalizationFactorSqr = GGXNormalizationFactor * GGXNormalizationFactor ;
    float AlphaPrimSqr = AlphaPrim * AlphaPrim;
    float SphereNormalization = AlphaPrimSqr / GGXNormalizationFactorSqr;
	float OneOverDenominator = 1. / ( (NDotH * NDotH) *(AlphaSqr - 1.0) + 1.0 );
	float Result = SphereNormalization * OneOverDenominator * OneOverDenominator;
    return(Result);
}

// Function 133
float lightI( float t )
{
    // fade light when close to floor to avoid harsh pop
    return clamp((light(t).y-FLOORY)/2.,0.,1.);
}

// Function 134
vec3 tower_lightcolor(vec2 coords)
{
    return mix( texture(iChannel0, (10./iChannelResolution[0].xy) * coords).rgb,
                texture(iChannel1, (10./iChannelResolution[1].xy) * coords).rgb,
                periodicsmoothstep( .05 * g_time - .005 * (coords.x + coords.y) - 1.));
}

// Function 135
vec3 getSkyLight(vec3 ro, vec3 rd, vec3 L, vec3 betaR, vec3 betaM, out vec3 miecolor, float _hm)
{
    vec3 light = vec3(0.0);
    float tmin = 0.0;
    float tmax = 0.0;
    float d = IntersectSphere(ro, rd, EARTHPOS, ATMOSPHERERADIUS, tmin, tmax);
    vec3 Pa = ro+rd*tmax;
    /*if(d>0.0)
        Pa = ro+rd*(tmax-tmin);*/
    float RdotL = dot(rd, L);
    float tCurrent = 0.0;
    float segL = tmax/16.0;
    /*if(d>0.0)
        segL = (tmax-tmin)/16.0;*/
    float g = 0.76; // 0.76
    float g2 = g*g;
    float hr = 7994.0; // 7994
    float hm = _hm;//800.0; // 1200
    // Rayleigh
    vec3 sumR = vec3(0.0);
    float phaseR = 3.0/(16.0*3.14)*(1.0+RdotL*RdotL);
    //vec3 betaR = vec3(5.5e-6, 13.0e-6, 22.4e-6);
    float opticalDepthR = 0.0;
    // Mie
    vec3 sumM = vec3(0.0);
    //float phaseM = 3.0/(8.0*3.14)*((1.0-g*g)+(1.0+RdotL*RdotL))/((2.0+g*g)+pow(1.0+g*g-2.0*g*RdotL, 1.5));
    // correction thanks to from http://http.developer.nvidia.com/GPUGems2/gpugems2_chapter16.html
    // my implementation has an error with the first terms in the equation
    float phaseM = (
        			(3.0*(1.0-g2))/
        			(2.0*(2.0+g2))
        			)*
        			(
                        (1.0+RdotL*RdotL)/
                        pow(1.0+g2-2.0*g*RdotL, 1.5)
                    );
    //vec3 betaM = vec3(21e-6);
    float opticalDepthM = 0.0;
    for (int i = 0; i < 16; ++i)
    {
        vec3 X = ro+rd*(tCurrent+0.5*segL);
        float h = length(X) - EARTHRADIUS;
        float _hr = exp(-h/hr)*segL;
        float _hm = exp(-h/hm)*segL;
        opticalDepthR += _hr;
        opticalDepthM += _hm;
        vec3 lRay = L;//normalize((SUNDIST*L)-X);
        float tlmin = 0.0;
        float tlmax = 0.0;
        float dl = IntersectSphere(X, L, EARTHPOS, ATMOSPHERERADIUS, tlmin, tlmax);
        float segLLight = tlmax/8.0;
        float tCurrentLight = 0.0;
        float opticalDepthLightR = 0.0;
        float opticalDepthLightM = 0.0;
        bool ended = true;
        for (int j = 0; j < 8; ++j)
        {
            vec3 samplePositionLight = X+L*(tCurrentLight + 0.5 * segLLight);
            float hLight = length(samplePositionLight) - EARTHRADIUS;
            if (hLight < 0.0)
            {
                ended = false;
                break;
            }
            opticalDepthLightR += exp(-hLight / hr) * segLLight;
            opticalDepthLightM += exp(-hLight / hm) * segLLight;
            tCurrentLight += segLLight;
        }
        if (ended)
        {
            vec3 tau = betaR * (opticalDepthR + opticalDepthLightR) + betaM * 1.1 * (opticalDepthM + opticalDepthLightM);
            vec3 attenuation = vec3(exp(-tau.x), exp(-tau.y), exp(-tau.z));
            sumR += _hr * attenuation;
            sumM += _hm * attenuation;
        }
        tCurrent += segL;
    }
    miecolor = SUNINTENSITY * sumM * phaseM;
    return SUNINTENSITY * (sumR * phaseR * betaR + sumM * phaseM * betaM);
}

// Function 136
vec3 WorldSpaceToDirectionalLightSpace (vec3 worldPosition)
{
	vec3 localPosition = (GetDirectionalLightMatrix() * vec4(worldPosition, 1.0)).xyz; 
    localPosition.xz /= directionalLightExtents;
    return localPosition;
}

// Function 137
float lightningNoise (vec2 forPos)
{
    forPos *= 4.0;
    forPos.y *= 0.85;
    float wobbleAmount1 = sin(forPos.y) * 0.5 + sin(forPos.y * 2.0) * 0.25 + sin(forPos.y * 4.0) * 0.125 + sin(forPos.y * 8.0) * 0.0625;
    float wobbleAmount2 = sin(forPos.x) * 0.5 + sin(forPos.x * 2.0) * 0.25 + sin(forPos.x * 4.0) * 0.125 + sin(forPos.x * 8.0) * 0.0625;
    float horizontalStrike = 1.0 - abs(sin(forPos.x + wobbleAmount1 * 1.1));
    float verticalStrike = 1.0 - abs(cos(forPos.y + wobbleAmount2 * 1.1));
    return (horizontalStrike + verticalStrike) * 0.35;
}

// Function 138
bool matIsLight( const in float mat ) {
    return mat < 0.5;
}

// Function 139
vec3 lightSample( const in LightInfo light, const in SurfaceInteraction interaction, out vec3 wi, out float lightPdf, float seed ) {
    vec2 u = vec2(random(), random());
    
    vec3 tangent = vec3(0.), binormal = vec3(0.);
    vec3 lightDir = normalize(light.position - interaction.point);
    createBasis(lightDir, tangent, binormal);
    
    float sinThetaMax2 = light.radius * light.radius / distanceSq(light.position, interaction.point);
    float cosThetaMax = sqrt(max(EPSILON, 1. - sinThetaMax2));
    wi = uniformSampleCone(u, cosThetaMax, tangent, binormal, lightDir);
    
    if (dot(wi, interaction.normal) > 0.) {
        lightPdf = 1. / (TWO_PI * (1. - cosThetaMax));
    }
    
	return light.L;
}

// Function 140
float DE_with_light(in vec3 p, in vec3 param)
{
    // floor and ceiling
    float d = min(p.y, -p.y+.2);

    // displaced by kaliset
	d -= kali_set_with_light(p*vec3(1,2,1), param).x;
    
    return d;
}

// Function 141
vec3 GetLightIntensity(){
	return AreaLightIntersity*LIGHTCOLOR;
}

// Function 142
vec3 light_distr(vec3 p)
{
    return vec3(1,1,1) * (4.*step(-light_sphere.w, -length(p - light_sphere.xyz)));
}

// Function 143
vec4 PointLight(int val, in IntersectResult ir){
    return vec4(pow(max(1.-min(1.,length(lights[val].xyz-ir.p)/lights[val].w),0.),3.));
}

// Function 144
vec3 GetLighting(Intersection i, vec3 origin)
{
	vec3 color = vec3(0,0,0);
	Light light;
	
	light.p = vec3(sin(iTime*0.3)*2.0,5,cos(iTime*0.3)*2.0+4.0);
	light.color = vec3(1,1,1);
	light.radius = 20.0;
	color += CalcLighting(light, i, origin);
	
	/*
	light.p = vec3(cos(time*0.2)*2.0,5,sin(time*0.2)*2.0+8.0);
	light.color = vec3(1,1,1);
	light.radius = 20.0;
	color += CalcLighting(light, i, origin);
*/
	
	return color;
}

// Function 145
vec3 calcLighting(vec3 col, vec3 ro, vec3 p, vec3 n, vec3 r, float sp) {
 
    vec3 lv = ro - p;
    vec3 ld = normalize(lv);
    
    float d = max(dot(ld,n),0.);
    float s = 0.;
    
    float atten = 1.-smoothstep(3., 25., length(lv));
    d *= atten;
    
    if(raymarch(p+ld*.1,ld) < length(lv))
        d = 0.;
    
    if(d > 0. && sp > 0.)
        s = pow(max(dot(ld,r),0.),sp) * atten;
    
    return col*d+s;
    
}

// Function 146
float de_light(vec3 p) {
    return length(p)-1.5;
}

// Function 147
vec3 lightItUp(vec3 hitPos, vec3 eyeDir, vec3 normal, Material material)
{
    const int numLights = 4;
    vec3 lightPositions[numLights] = vec3[numLights](
        vec3(-10.0, 10.0, 10.0),
        vec3(sin(iTime) * 10.0, 10.0, cos(iTime) * 10.0),
        vec3(16.0, -10.0, 7.0),
        vec3(-7.0, -15.0, 20.0)
    );
    vec3 lightColors[numLights] = vec3[numLights](
        vec3(1.0),
        vec3(1.0, 0.2, 0.02),
        vec3(0.1, 0.8, 1.0),
        vec3(1.0)
    );
    
    vec3 ret = material.emissive;
    
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, material.albedo, material.metallic);
    
    for (int i = 0; i < numLights; i++)
    {
        vec3 lightVec = normalize(lightPositions[i] - hitPos);
        vec3 reflectionVec = reflect(eyeDir, normal);
     
        // calculate per-light radiance
        vec3 N = normal;
        vec3 WorldPos = hitPos;
        vec3 V = -eyeDir;
        vec3 L = lightVec;
        vec3 H = normalize(V + L);
        float distance    = length(lightPositions[i] - WorldPos);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance     = lightColors[i];// * attenuation;        
        
        // cook-torrance brdf
        float NDF = DistributionGGX(N, H, material.roughness);        
        float G   = GeometrySmith(N, V, L, material.roughness);      
        vec3 F    = vec3(fresnelSchlick(max(dot(H, V), 0.0), F0));
        
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - material.metallic;	  
        
        vec3 nominator    = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001; 
        vec3 brdf = nominator / denominator;

        // add to outgoing radiance Lo
        float NdotL = max(dot(N, L), 0.0);      
        float PI = 3.14159265;
        vec3 lightContribution = (kD * material.albedo / PI + brdf) * radiance * NdotL; 

        ret += lightContribution;
    }
    
    return ret;
}

// Function 148
vec4 NormalPointLight(int val, in IntersectResult ir){
    vec3 v = lights[val].xyz-ir.p;
    return vec4(1.)*max(0.,dot(ir.n,normalize(v)))*pow(max(1.-min(1.,length(v)/lights[val].w),0.),3.);
}

// Function 149
vec3 PBRLight(vec3 pos, vec3 normal, vec3 view, PBRMat mat, vec3 lightPos, vec3 lightColor, float fresnel, MatSpace ps, bool shadows)
{
    //Basic lambert shading stuff
    
    //return vec3(fresnel);
    
    vec3 key_Dir = lightPos - pos;
    
    float key_len = length(key_Dir);
    

    
    key_Dir /= key_len;
    

    float key_lambert = max(0.0, dot(normal, key_Dir));
    
     
    float key_shadow = shadows ? S(0.0, 0.10, shadow(pos, normal, lightPos, ps)) : 1.0; 
    
    float diffuseRatio = key_lambert * key_shadow;
   
    
    vec3 key_diffuse = vec3(diffuseRatio);
    

    // The more metalness the more present the Fresnel
    float f = pow(fresnel + 0.5 * mat.metalness, mix(2.5, 0.5, mat.metalness));
    
    // metal specular color is albedo, it is white for dielectrics
    vec3 specColor = mix(vec3(1.0), mat.albedo, mat.metalness);
    
    vec3 col = mat.albedo * key_diffuse * (1.0 - mat.metalness);
    
    // Reflection vector
    vec3 refDir = reflect(view, normal);
    
    // Specular highlight (softer with roughness)
    float key_spec = max(0.0, dot(key_Dir, refDir));
    key_spec = pow(key_spec, 10.0 - 9.0 * mat.roughness) * key_shadow;
    
    float specRatio = mat.metalness * diffuseRatio;
    
    col += vec3(key_spec) * specColor * specRatio;
    col *= lightColor;
    

    
    return col;
}

// Function 150
void constructLightPath( inout float seed ) {
    vec3 ro = randomSphereDirection( seed );
    vec3 rd = cosWeightedRandomHemisphereDirection( ro, seed );
    ro = lightSphere.xyz - ro*lightSphere.w;
    vec3 color = LIGHTCOLOR;
 
    for( int i=0; i<LIGHTPATHLENGTH; ++i ) {
        lpNodes[i].position = lpNodes[i].color = lpNodes[i].normal = vec3(0.);
    }
    
    bool specularBounce;
    float w = 0.;
    
    for( int i=0; i<LIGHTPATHLENGTH; i++ ) {
		vec3 normal;
        vec2 res = intersect( ro, rd, normal );
        
        if( res.y > 0.5 && dot( rd, normal ) < 0. ) {
            ro = ro + rd*res.x;            
            color *= matColor( res.y );
            
            lpNodes[i].position = ro;
            if( !matIsSpecular( res.y ) ) lpNodes[i].color = color;// * clamp( dot( normal, -rd ), 0., 1.);
            lpNodes[i].normal = normal;
            
            rd = getBRDFRay( normal, rd, res.y, specularBounce, seed );
        } else break;
    }
}

// Function 151
vec3 lightColor()
{
    return vec3(1.0, 0.9, 0.75) * 3.9;
}

// Function 152
vec3 calcLighting(Hit scn, Ray ray, vec3 n, bool shadowOnly) {

	float diff = max(dot(LIGHT_DIR,n), 0.);
	if(shadowOnly) diff = 1.;
    
	if(scn.id == 1) {
	
		Ray sr = Ray(scn.p + LIGHT_DIR * .003, LIGHT_DIR);
		Hit sh = raymarch(sr);
	
		if(sh.id == 0) {
		
            vec4 t = shadeJCVD(sh,sr,calcNormal(scn.p));    
			diff = mix(diff,0.,t.a);
		
		}
	
	}

    return vec3(diff);
	return (shadowOnly ? vec3(1.) : vec3(1.,.98,.9)) * diff;

}

// Function 153
vec3 SampleLightsInScene(
	vec3 P,
	vec3 N,
	bool haltonEnabled,
	vec2 offset,
	int objId,
	int frameIndex, // iFrame
	int numLightSamples,
	inout uint wasSampled)
{
	bool sphericalLightIsTextured = false; // enable this if SampleLightColor needs position for spherical lights
	vec3 e = vec3(0);
	for (int lightId = 0; lightId < NO_UNROLL_(NUM_OBJECTS, objId); lightId++) {
		Object light = objects[lightId];
		if (IsQuad(light) && dot(light.pos - P, light.quadNormal) >= 0.0) { // facing away?
			wasSampled |= (1U << lightId); // might as well mark this light as sampled, we won't hit it in the next bounce
			continue;
		}
		if (ShouldSampleLight(P, N, light, e)) {
			vec3 l = vec3(0);
			if (MIS_enabled) {
				float q = MIS_ratio_default; // controls ratio N_h / N_d (hemisphere samples to direct light samples)
				if (MIS_light_dist_enabled && MIS_light_dist_max > 0.0) {
					float dmax = MIS_light_dist_max;
					float dmin = min(dmax - 0.0001, MIS_light_dist_min);
					float ds = 1.0/(dmin - dmax);
					float d0 = -ds*dmax;
					q = clamp(DistanceToObject(P, light)*ds + d0, 0.0, 1.0);
				}
				if (MIS_light_dist_dbg)
					e += vec3(q*q);
				int N_h = int(floor(0.5 + float(numLightSamples)*q)); // [0..numLightSamples]
				int N_d = numLightSamples - N_h; // [0..numLightSamples]
				for (int i = 0; i < N_d; i++) {
					vec2 s = haltonEnabled ? fract(offset + Halton23(i + frameIndex*N_d)) : rand2(seed);
					l += SampleLightMIS_d(light, lightId, objId, P, N, s, float(N_d), float(N_h), sphericalLightIsTextured);
				}
				for (int i = 0; i < N_h; i++) {
					vec2 s = haltonEnabled ? fract(offset + Halton23(i + frameIndex*N_h)) : rand2(seed);
					l += SampleLightMIS_h(light, lightId, objId, P, N, s, float(N_d), float(N_h));
				}
			} else {
				for (int i = 0; i < NO_UNROLL_(numLightSamples, objId); i++) {
					vec2 s = haltonEnabled ? fract(offset + Halton23(i + frameIndex*numLightSamples)) : rand2(seed);
					l += SampleLight(light, lightId, objId, P, N, s, sphericalLightIsTextured);
				}
				l /= float(numLightSamples);
			}
			e += l;
			wasSampled |= (1U << lightId);
		}
	}
	return e;
}

// Function 154
vec3 computeLight(vec3 pos, vec3 color, vec3 normal) {
    
    vec3 toLight1 = LIGHT1_POS - pos;
    vec3 ambient_light = vec3 (0, 0, 0);
    float distSq1 = dot(toLight1, toLight1);
    float att1 = isOccluded(pos, LIGHT1_POS + randDir * LIGHT1_RADIUS) ? 0.0 : 20.0 / distSq1;
    
    vec3 toLight2 = LIGHT2_POS - pos;
    float distSq2 = dot(toLight2, toLight2);
    float att2 = isOccluded(pos, LIGHT2_POS + randDir * LIGHT2_RADIUS) ? 0.0 : 10.0 / distSq2;
    
    return ambient_light + color * (
        max(0.0, dot(normal, normalize(toLight1))) * att1 * LIGHT1_COLOR
        + max(0.0, dot(normal, normalize(toLight2))) * att2 * LIGHT2_COLOR
        + texture(iChannel1, normal).rgb * 0.2
    );
}

// Function 155
float light_pdf( const in LightInfo light, const in SurfaceInteraction interaction ) { 
    float sinThetaMax2 = light.radius * light.radius / distanceSq(light.position, interaction.point);
    float cosThetaMax = sqrt(max(EPSILON, 1. - sinThetaMax2));
    return 1. / (TWO_PI * (1. - cosThetaMax)); 
}

// Function 156
bool ShouldSampleLight(vec3 P, vec3 N, Object light, inout vec3 dbg)
{
	if (IsLight(light)) {
		if (direct_light_dist_enabled && direct_light_dist_max <= 0.0)
			return true;
		else {
			float dmax = direct_light_dist_max;
			float dmin = min(dmax - 0.0001, direct_light_dist_min);
			float ds = 1.0/(dmin - dmax);
			float d0 = -ds*dmax;
			float q = clamp(DistanceToObject(P, light)*ds + d0, 0.0, 1.0); // q=1 @ min dist, q=0 @ max dist
			if (direct_light_dist_dbg) {
				dbg += vec3(q*q);
				return true;
			} else if (q < rand(seed))
				return true;
		}
	}
	return false;
}

// Function 157
vec3 lighting(in vec3 position, in vec3 normal, in vec3 viewDir, in Material material, in float shadow)
{
    vec3 lightDir = lightDirection();
    vec3 lightCol = lightColor();
    
    // Specular
    float ndoth = dot(normal, normalize(-viewDir - lightDir));
    float specularBlinnPhong = pow(max(ndoth, 0.0), material.roughness);
    
    // Diffuse
    float ndotl = dot(normal, -lightDir);
    float diffuseLambert = max(0.0, ndotl);
    
    // Final terms
    float ambient = 0.4;
    float diffuse = diffuseLambert * shadow;
    float specular = specularBlinnPhong * material.shininess * diffuse;
    
    return (ambient * material.albedo + (diffuse * material.albedo + specular))* lightCol;
}

// Function 158
float fancyLight(ray primaryRay, vec4 light) {
	float luminance = 0.0;
	//vector from origin to light center
	vec3 originToLight = light.xyz - primaryRay.origin;
	
	//check to see if the light is behind us
	if(dot(primaryRay.direction, originToLight) >= 0.0) {
		//see if our ray is within the light boundary
		vec3 nearest = -(originToLight
						 + primaryRay.direction * dot(-originToLight, primaryRay.direction));
		float dist = length(nearest);
		if(dist <= light.w) {
			float lightness = (light.w - dist) / light.w;
			luminance = pow(lightness, 22.0);
		}
	}
	return luminance;
}

// Function 159
vec3 LightVaktTorn(vec3 RP, vec3 D, vec3 cRET) { //45, 40, 45
    vec3 RET=cRET; vec3 LRET;
    LRET=LightRing(vec3(22.5,27.,22.5),8.,RP,D,3.,2.,0.25,0); if (LRET.z<RET.z) RET=LRET; //Överst
    return RET;
}

// Function 160
float specularLight (vec3 p, vec3 n, vec3 r){
    vec3 nr = reflect(r, n);
    return pow(dot(nr, -r), 2.);
}

// Function 161
vec3 
neighbor_light( vec3 hp, vec3 n, vec3 neighbor_cell_coord)
{
   vec3 neighbor_color = tower_lightcolor(neighbor_cell_coord.xz);
   vec3 light_pos = tower_lightpos(neighbor_cell_coord.xz, neighbor_color);
   light_pos += g_cellsize * vec3(neighbor_cell_coord.x + .5, 0., neighbor_cell_coord.z + .5);
   vec3 l = hp - light_pos;        
   float llen = length(l);
   return neighbor_color * max(0., dot(-normalize(l), n)) * pow(1./llen, .3);
}

// Function 162
float GetLight(vec3 p){
    //position of the light source
    
    
    //lightPos.xz += vec2(sin(iTime),cos(iTime));
    //light vector
    vec3 l = normalize(lightPos-p);
    
    //normal of object
    vec3 n = GetNormal(p);
    
    // dot product of the light vector and normal of the point
    // will give us the amount of lighting to apply to the point
    // dot() evaluates to values between -1 and 1, so we will clamp it
    float diff = clamp(dot(n, l),0.,1.);
    
    // calculate if point should be a shadow:
    // raymarch from point being calculated towards light source
    // if hits surface of something else before the light,
    // then it must be obstructed and thus is a shadow
    // the slight offset "p+n*SURFACE_DIST*1.1" is needed to ensure the
    // break condistions in the function are not met too early
    float d = RayMarch(p+n*SURFACE_DIST*1.1,l).x;
    if(d < length(lightPos-p)){
        diff *= .1;
    }
    
    return diff;
}

// Function 163
vec3 computeEnlighting( in vec3 cameraPos, in vec3 cameraDir, in vec3 lightDir ) {

	cameraDir += perturb3(cameraDir,.06,1.5);
	// position of I : point at the surface of the sphere
	float a = dot(cameraDir,cameraDir);
	float b = 2.0*dot(cameraDir,cameraPos);
	float c = dot(cameraPos,cameraPos) - rad*rad;
	float delta = b*b-4.0*a*c;
		
	if (delta <= 0.0)
		return skyColor;
		
	float d1 = (-b + sqrt(delta))/(2.0*a);
	float d2 = (-b - sqrt(delta))/(2.0*a);
	
	vec3 posI = cameraPos + d1 * cameraDir;
	vec3 posIprim = cameraPos + d2 * cameraDir;
	float d3 = length(posI-posIprim); // length of the path without scattering
	
	// normal of the plane containing the camera & the light
	vec3 n = cross(-lightDir,-cameraDir);
	n = normalize(n);	
	
	float d = dot(posI,n); // distance plane - center of the sphere
	vec3 C = n*d; // center of the circle
	float r = clamp(length(posI-C),0.001,rad-0.001); // radius of the circle
	
	float theta = acos(clamp(dot(normalize(cameraDir),normalize(C-posI)),-1.,1.));
	float y = r*sin(theta);
	
	// projection of lightDir
	float IPS = acos(clamp(dot(normalize(-cameraDir),normalize(lightDir)),-1.,1.));
	
	vec2 L = vec2(-cos(IPS),sin(IPS));

	// check the orientation
	if (dot(cross(cameraDir,-lightDir),cross(cameraDir,normalize(posI-C))) > 0.0) {
		L.y = -L.y;
	}
	
	// rayleigh diffusion function
	float rayleigh = cos(IPS)*cos(IPS)+1.0; 

	vec3 transmittance = sphericalTransmittanceGradient(L, r, r-y,length(C))*rayleigh;
	transmittance *= sunColor;
	transmittance += exp(-computeMeanDensRay(y, length(C), r))*skyColor; 
	return transmittance;
}

// Function 164
float MapStreeLight(vec3 p)
{
  float d= fCylinder(p-vec3(0.31, -3.5, 0.), 0.7, 0.01);
  d=fOpPipe(d, fCylinder(p-vec3(.31, -4., 0.), 0.7, 3.0), .05);   
  d=min(d, fCylinderH(p-vec3(.98, -6.14, 0.), 0.05, 2.4));        
  d=fOpUnionChamfer(d, fCylinderH(p-vec3(.98, -8., 0.), 0.1, 1.0), 0.12);  
  d=min(d, sdSphere(p-vec3(-0.05, -3.4, 0.), 0.2));  
  d=min(d, sdSphere(p-vec3(-0.05, -3.75, 0.), 0.4));        
  d=max(d, -sdSphere(p-vec3(-.05, -3.9, 0.), 0.45)); 

  return d;
}

// Function 165
bool isLightVisible( Ray shadowRay ) {
    float distToHit;
    SurfaceHitInfo tmpHit;
    
    raySceneIntersection( shadowRay, EPSILON, true, tmpHit, distToHit );
    
    return ( tmpHit.material_id_ == MTL_LIGHT );
}

// Function 166
void aurorasWithLightningNoise( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 q = fragCoord.xy / iResolution.xy;
    vec2 p = q - 0.5;
	p.x*=iResolution.x/iResolution.y;
    
    vec3 ro = vec3(0,0,-6.7);
    vec3 rd = normalize(vec3(p,1.3));
    vec2 mo = iMouse.xy / iResolution.xy-.5;
    mo = (mo==vec2(-.5))?mo=vec2(-0.1,0.1):mo;
	mo.x *= iResolution.x/iResolution.y;
    rd.yz *= mm2(mo.y);
    rd.xz *= mm2(mo.x + sin(time*0.05)*0.2);
    
    vec3 col = vec3(0.);
    vec3 brd = rd;
    float fade = smoothstep(0.,0.01,abs(brd.y))*0.1+0.9;
    
    col = bg(rd)*fade;
    
    if (rd.y > 0.){
        vec4 aur = smoothstep(0.,1.5,aurora(ro,rd))*fade;
        col += stars(rd);
        col = col*(1.-aur.a) + aur.rgb;
    }
    else //Reflections
    {
        rd.y = abs(rd.y);
        col = bg(rd)*fade*0.6;
        vec4 aur = smoothstep(0.0,2.5,aurora(ro,rd));
        col += stars(rd)*0.1;
        col = col*(1.-aur.a) + aur.rgb;
        vec3 pos = ro + ((0.5-ro.y)/rd.y)*rd;
        float nz2 = domainWarp(pos.xz*vec2(.5,.7));
        col += mix(vec3(0.2,0.25,0.5)*0.08,vec3(0.3,0.3,0.5)*0.7, nz2*0.4);
    }
    
	fragColor = vec4(col, 1.);
}

// Function 167
float lightPdf(const in vec4 light, const in SurfaceInteraction interaction) {
	float sinThetaMax2 = light.w * light.w / distanceSq(light.xyz, interaction.point);
    float cosThetaMax = sqrt(max(EPSILON, 1. - sinThetaMax2));
    return 1. / (TWO_PI * (1. - cosThetaMax));
}

// Function 168
vec3 DrawPointLight(vec3 ro, vec3 rd, float d, vec3 lp, vec3 lc, float r)
{
	vec3 res = vec3(0);
	float pld = sphDistance( ro, rd, vec4(lp, r) );
	float plhit = shpIntersect( ro, rd, vec4(lp, r) );
	if (plhit > 0.0)
	{
		vec3 p = ro+rd*pld;
		float len = length(lp-p);
			
		if (d>len)
		{
			vec3 k = rd - normalize(lp-ro);
			res += lc * pointLight * (1.-pld / dot(k, k));
		}
	}
	return res;
}

// Function 169
vec3 calculateDirectLight(const in LightInfo light, const in SurfaceInteraction interaction, const in MaterialInfo material, out vec3 wi, out vec3 f, out float scatteringPdf) {
	
    // Light MIS
    vec3 wo = -interaction.incomingRayDir;
    vec3 Ld = vec3(0.);
    float lightPdf = 0., visibility = 1.;
    scatteringPdf = 0.;
    bool isBlack = false;
    
    vec3 Li = sampleLightType( light, interaction, wi, lightPdf, visibility, seed );
    Li *= visibility;
    
    isBlack = dot(Li, Li) == 0.;
    
    if (lightPdf > EPSILON && !isBlack ) {
        vec3 f = bsdfEvaluate(wi, wo, interaction.tangent, interaction.binormal, interaction, material) * abs(dot(wi, interaction.normal));
        float weight = 1.;
        
        #ifdef USE_MIS
        	scatteringPdf = bsdfPdf(wi, wo,interaction.tangent, interaction.binormal, interaction, material);
            weight = powerHeuristic(1., lightPdf, 1., scatteringPdf);
        #endif
        
        if( light.type == LIGHT_TYPE_SUN )
            weight = 1.;
            
		isBlack = dot(f, f) == 0.;
        if (!isBlack) {
           Ld += Li * f * weight/ lightPdf;
        }
    }
    
    // BSDF MIS
    f = bsdfSample( wi, wo, interaction.tangent, interaction.binormal, scatteringPdf, interaction, material);
    f *= abs(dot(wi, interaction.normal));
	
    #ifdef USE_MIS
        isBlack = dot(f, f) == 0.;
        Li = light.L;

        if (!isBlack && scatteringPdf > EPSILON && light.type != LIGHT_TYPE_SUN) {
            float weight = 1.;

            lightPdf = light_pdf(light, interaction);
            if (lightPdf < EPSILON) return Ld;
            weight = powerHeuristic(1., scatteringPdf, 1., lightPdf);

            Li *= visibilityTest(interaction.point + wi * .01, wi);
            isBlack = dot(Li, Li) == 0.;
        	if (!isBlack) {
            	Ld +=  Li * f * weight / scatteringPdf;
            }
        }
    #endif
    return Ld;
}

// Function 170
vec4 applyLighting(vec4 inpColor, vec2 uv, vec3 normal, vec3 LightPos, vec4 LightColor, vec4 AmbientColor)
{
   // if(distance(uv.xy, LightPos.xy) < 0.01) return vec4(1.,0.,0.,1.);
    vec3 LightDir = vec3(LightPos.xy - uv, LightPos.z);
    vec3 N = normalize(normal);
    vec3 L = normalize(LightDir);
    vec3 Diffuse = (LightColor.rgb * LightColor.a) * max(dot(N, L), 0.0);
    vec3 Ambient = AmbientColor.rgb * AmbientColor.a;
    vec3 Intensity = Ambient + Diffuse;
    vec3 FinalColor = inpColor.rgb * Intensity;
    return vec4(FinalColor, inpColor.a);
}

// Function 171
vec3 getLighting( vec3 p, vec3 normal ) {
    vec3 l = vec3(0.);
    
    float i = getSphereLightIntensity(0.);
    if (i > 0.) {
	    l += sphereCol(time) * (i * getLightIntensity(p, normal, sphereCenter(activeSpheres[0]), .375));
    } else {    
        i = getSphereLightIntensity(1.);
        if (i > 0.) {
            l += sphereCol(time+1.) * (i * getLightIntensity(p, normal, sphereCenter(activeSpheres[1]), .25));
        }
    }
    
    vec3 robot = mix(sphereCol(time), sphereCol(time-1.), getSphereLightIntensity(0.));
    vec3 lp = rotateY(vec3(joints[2].x, joints[2].y+1.,0), -jointYRot);
    i = getLightIntensity(p, normal, lp, .5);
    i += getLightIntensity(p, normal, vec3(0,2,0), .25);
    l += i * robot;
    
    return l;
}

// Function 172
vec2 ComputeLightmapUV(Object obj, vec3 V, sampler2D lightmapSampler)
{
    vec2 lightmapUV = vec2(-1);
    vec2 lightmapResInv = 1.0/vec2(textureSize(lightmapSampler, 0)); // we could pass this in ..
    if (IsQuad(obj)) {
        vec2 st;
        st.x = dot(V, obj.quadBasisX);
        st.y = dot(V, obj.quadBasisY);
        vec2 uv = st*0.5 + vec2(0.5); // [0..1]
        vec4 atlasBounds = obj.lightmapBounds;
        atlasBounds.zw -= atlasBounds.xy; // width, height
        if (float(LIGHTMAP_QUAD_INSET) < 0.5) { // don't sample outside the lightmap bounds (if we are filtering)
            vec2 uvmin = vec2(0.5)/atlasBounds.zw;
            vec2 uvmax = vec2(1) - uvmin;
            uv = clamp(uv, uvmin, uvmax);
        }
        atlasBounds *= lightmapResInv.xyxy;
        lightmapUV = atlasBounds.xy + atlasBounds.zw*uv;
    } else {
        int faceRow;
        int faceCol;
        vec2 facePos;
        vec3 Va = abs(V);
        float Vamax = max(max(Va.x, Va.y), Va.z);
        if (Vamax == Va.x) {
            faceCol = 0;
            faceRow = V.x < 0.0 ? 1 : 0;
            facePos = V.yz/Va.x;
        } else if (Vamax == Va.y) {
            faceCol = 1;
            faceRow = V.y < 0.0 ? 1 : 0;
            facePos = V.zx/Va.y;
        } else { // Vamax == Va.z
            faceCol = 2;
            faceRow = V.z < 0.0 ? 1 : 0;
            facePos = V.xy/Va.z;
        }
        vec2 faceUV = facePos*0.5 + vec2(0.5); // [0..1]
        float faceRes = GetLightmapSphereFaceRes(obj);
        vec2 faceBoundsMin = vec2(faceCol + 0, faceRow + 0)*faceRes + vec2(LIGHTMAP_SPHERE_FACE_INSET);
        vec2 faceBoundsMax = vec2(faceCol + 1, faceRow + 1)*faceRes - vec2(LIGHTMAP_SPHERE_FACE_INSET);
        vec2 uv = obj.lightmapBounds.xy + faceBoundsMin + (faceBoundsMax - faceBoundsMin)*faceUV;
        lightmapUV = uv*lightmapResInv;
    }
    return lightmapUV;
}

// Function 173
vec3 getLighting( in vec3 p, in vec3 dir, float index ) {
    // get surface albedo and roughness
    vec4 albedo = vec4(0);
    float d = de(p, true, albedo);
    // get surface normal
    vec3 n = getNormal(p, d);
    
    vec3 result = vec3(0);
    const vec3 sunDir = normalize(vec3(1, 4, 2));
    const vec3 subDir = normalize(vec3(2, -7, 3));
    
    // add two lights, main one with shadows
    float shadow = traceShadow(p+n*0.01, sunDir, n, 0.05);
    result += computeLighting(n, dir, albedo.rgb, (1.0-albedo.a)*0.3, albedo.a, sunDir,
                              vec3(0.9, 0.85, 0.5)*10.0)*shadow;
    result += computeLighting(n, dir, albedo.rgb, (1.0-albedo.a)*0.3, albedo.a, subDir,
                              COLOR_SUB*0.5);
    // and add subsurface scattering
    result += computeSSS(n, dir, albedo.rgb, albedo.a, index, sunDir,
                         vec3(0.9, 0.85, 0.5)*10.0);
    result += computeSSS(n, dir, albedo.rgb, albedo.a, index, subDir,
                         COLOR_SUB*0.5);
    
    return result;
}

// Function 174
vec3 lightpos(vec3 world)
{
    //vec3 lps = vec3(10.0,0.0,0.0) * zrot(iTime);
    vec3 lps = vec3(10.0,4.0,-2.5);
    return lps;
}

// Function 175
vec3 sampleLight( const in vec3 ro, inout float seed ) {
    vec3 n = randomSphereDirection(seed) * lightSphere.w;
    return lightSphere.xyz + n;
}

// Function 176
void initLights()
{
    vec3 col = vec3(0.925, 0.968, 0.972);
    
    ambientcolor = col;
    
    // Center Up
    lights[0].pos = vec3(0., 13.0, 0.);
    lights[0].color = col*0.25;
    
    // Window
    lights[1].pos = vec3(14.0, 15.5, -12.0);
    lights[1].color = col*0.25;
    
    lights[2].pos = vec3(14.0, 15.5, -6.0);
    lights[2].color = col*0.25;
}

// Function 177
vec3 light(in vec3 pos, in vec3 nor)
{
	vec3 ro = pos + 0.001 * nor;
	vec3 ret = vec3(0.0);

	for (int i = 0; i < ld.length(); i++)
	{
		if (isLightVisible(ro, i))
		{
			float att = attenuation(pos, ld[i].pos, ld[i].r);
			ret = ret + ld[i].col * att;
		}
	}

	return ret;
}

// Function 178
float light_time_per_m()
{
    return (iMouse.z > 0.) ? (min(1.,max((MOUSEY-0.25)/0.7,0.))*0.06) : (sin(iTime*.25)*.5+.5)*.06 ;
}

// Function 179
bool IsLight(Object obj)
{
    return max(max(obj.emissive.x, obj.emissive.y), obj.emissive.z) > 0.0;
}

// Function 180
vec3 LightRing(vec3 cp, float r, float angle, vec3 p, vec3 dir, float N, float R, float I, int Type) {
    //N lights with size R (pixels) and intensity I on a circle with normal e_y, radius r and center p
    vec3 Int=vec3(0.); vec2 Intensity;
    if (dir.y==0.) {
        //Projection: line
    } else {
        //Projection: ellips
        vec2 CircleCoord=(p-cp+dir*(-(p.y-cp.y)/dir.y)).xz;
        float Angle=atan(CircleCoord.x,CircleCoord.y)+RADIAN*0.5;
        float FAngle=floor(Angle*IRADIAN*N)*(RADIAN/N)-angle;
        vec3 LightPos=cp+vec3(-r*sin(FAngle),0.,-r*cos(FAngle));
        Intensity=((TraceSphere(p,dir,LightPos,R*IRES.y*(LightPos.z-p.z))<0.)?
                                    vec2(0.,10000.):vec2(I,LightPos.z-p.z));
        FAngle+=(RADIAN/N);
        LightPos=cp+vec3(-r*sin(FAngle),0.,-r*cos(FAngle));
        Intensity=((TraceSphere(p,dir,LightPos,R*IRES.y*(LightPos.z-p.z))<0.)?
                                    Intensity:vec2(I,LightPos.z-p.z));
        Int[Type]=Intensity.x; Int.z=Intensity.y;
    }
    return Int;
}

// Function 181
vec3 sampleLight( const in vec3 ro, inout float seed ) {
    vec3 n = randomSphereDirection( seed ) * lightSphere.w;
    return lightSphere.xyz + n;
}

// Function 182
vec3 calculateDirectLight(const in LightInfo light, const in SurfaceInteraction interaction, const in MaterialInfo material, out vec3 wi, out vec3 f, out float scatteringPdf) {
    
        
    vec3 wo = -interaction.incomingRayDir;
    vec3 Ld = vec3(0.);
    float lightPdf = 0., visibility = 1.;

    vec3 Li = sampleLightType( light, interaction, wi, lightPdf, visibility, seed, material);
    Li *= visibility;

    f = bsdfEvaluate(wi, wo, interaction.tangent, interaction.binormal, interaction, material) * abs(dot(wi, interaction.normal));        
    Ld += Li * f;

    return Ld;
}

// Function 183
vec3 GetVolumetricLighting(Ray ray, float maxDist, vec2 fragCoord)
{
	vec3 color = vec3(0,0,0);
	Light light;
	light.p = vec3(sin(iTime*0.3)*2.0,5,cos(iTime*0.3)*2.0+4.0);
	light.color = vec3(1,1,1);
	light.radius = 20.0;
	
	float inscattering = maxDist/200.0;
	float volRayStep = maxDist/float(VOLUMETRIC_SAMPLES-1);
	float randomStep = rand(fragCoord.xy)*volRayStep;
	Ray volRay;
	volRay.o = ray.o + ray.dir*randomStep;
	for(int v = 0; v < VOLUMETRIC_SAMPLES; v++)
	{
		vec3 lightVec = light.p-volRay.o;
		float lightDist = length(lightVec);
		volRay.dir = lightVec/lightDist;
		Intersection i = SceneIntersection(volRay);
		if(i.dist > lightDist)
		{
			color += CalcIrradiance(light, volRay.o)*inscattering;
		}
		volRay.o += ray.dir * volRayStep;
	}
	
	return color;
}

// Function 184
vec3 computeLighting(in vec3 normal, in vec3 viewDir,
                     in vec3 albedo, in float metallic, in float roughness,
                     in vec3 lightDir, in vec3 radiance) {
    
    vec3 result = vec3(0);
    
    // find half way vector
    vec3 halfwayDir = normalize(viewDir + lightDir);
    
    // figure out surface reflection
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);
    
    // find the PBR terms
    float NDF = DistributionGGX(normal, halfwayDir, roughness);
    float G = GeometrySmith(normal, viewDir, lightDir, roughness);
    vec3 F = fresnelSchlick(max(dot(halfwayDir, viewDir), 0.0), F0);
    
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;
    
    // Cook Torrance BRDF
    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(normal, viewDir), 0.0) * max(dot(normal, lightDir), 0.0);
    vec3 specular = numerator / max(denominator, 0.001);  
    
    // add light contribution
    float NdotL = max(dot(normal, lightDir), 0.0);
    result += (kD * albedo / PI + specular) * radiance * NdotL;
    
    return result;
}

// Function 185
vec3 ComputeLight(vec3 P, vec3 D, vec3 N, vec3 LD, vec2 rand, float iTime) {
    HIT LHIT; mat3 NM; vec3 Sample; vec3 Color=vec3(0.);
    //Diffuse
    NM=TBN(N);
    Sample=RandSample(rand)*NM;
    if (TraceRay(P,Sample,LHIT,128.,iTime)) {
        if (LHIT.Mat==2.) Color+=LHIT.C; else {
            //Direct Light
            float LDot=dot(LD,LHIT.N);
            if (LDot>0.) {
                NM=TBN(LD);
                Sample=LD;
                if (!TraceRay(LHIT.P+LHIT.N*0.05,Sample,LHIT,128.,iTime))
                    Color.xyz+=SunColor*LDot*LHIT.C*1.5;
            }
        }
    } else
        Color.xyz+=(1.-0.5*Sample.y)*SkyColor;
    //Return
    return Color;
}

// Function 186
vec4 doLighting(vec3 eyePoint, vec3 objPoint, vec3 normalAtPoint, vec3 lightPos) {
	float fresnelBias = 0.25;
	float fresnelPower = 5.0;
	float fresnelScale = 1.0;
	float shininess = 20.0;
	vec4 lightParams = vec4(fresnelBias, fresnelPower, fresnelScale, shininess);
	return doLighting(eyePoint, objPoint, normalAtPoint, lightPos, lightParams); 
}

// Function 187
void get_cam_and_light(
    in float time,
    out vec3 camera_pos, out vec3 camera_look_at, out vec3 camera_up,
    out float fovy_deg, out vec3 light_pos)
{
    camera_pos = pos_clelies(time, ORBIT_RADIUS);
    camera_look_at = vec3(0.0);
    camera_up = vec3(0.0, 1.0, 0.0);
    fovy_deg = FOVY_DEG;
    light_pos = camera_pos;
}

// Function 188
vec3 BRDFLightSample(in Intersection intersecNow,out Intersection intersecNext,out vec3 wi,out float pdf){
	vec3 Li = vec3(0.);
    float x1 = GetRandom(),x2 = GetRandom();
    wi = sample_uniform_hemisphere(intersecNow.normal,x1,x2,pdf);
    Ray shadowRay = Ray(intersecNow.surface,wi);
    SceneIntersect(shadowRay, intersecNext);
    return Li;
}

// Function 189
float sphLight( vec3 P, vec3 N, vec4 L)
{
  vec3 oc = L.xyz  - P;
  float dst = sqrt( dot( oc, oc ));
  vec3 dir = oc / dst;
  
  float c = dot( N, dir );
  float s = L.w  / dst;
    
  return max(0., c * s);
}

// Function 190
void initLightSphere( float time ) {
	lightSphere = vec4( 3.0+2.*sin(time),2.8+2.*sin(time*0.9),3.0+4.*cos(time*0.7),0.5);
}

// Function 191
float lightTrace(vec2 o, vec2 r, float maxDst){
    
    // Raymarching.
    float d, t = 0.;
    
    
    // 96 iterations here: If speed and complilation time is a concern, choose the smallest 
    // number you can get away with. Apparently, swapping the zero for min(0, frame) can
    // force the compliler to not unroll the loop, so that can help sometimes too.
    for(int i=0; i<16; i++){
        
        // Surface distance.
        d = map(o + r*t);
        
        // In most cases, the "abs" call can reduce artifacts by forcing the ray to
        // close in on the surface by the set distance from either side.
        if(d<0. || t>maxDst) break;
        
        
        // No ray shortening is needed here, and in an ideal world, you'd never need it, but 
        // sometimes, something like "t += d*.7" will be the only easy way to reduce artifacts.
        t += d*RSF_SHAD;
    }
    
    //t = min(t, maxDst); // Clipping to the far distance, which helps avoid artifacts.
    
    return t;
    
}

// Function 192
vec3 lighting(in Hit h) {
  if (h.ray.len > MAX_TRACE_DIST) return pow(textureLod(iChannel0, -h.ray.dir, 0.).rgb,vec3(0.4));
  vec4 fgi = gi(h.pos, h.nml);    // Fake Global Illumination
  vec4 fcs = cs(h.pos, dif.dir);  // Fake Caustic Shadow
  //   lin = ([Ambient]        + [Diffuse]        * [SS] + [CAUSTICS]) * [AO] + [GI]
  vec3 lin = (_lit(h.nml, amb) + _lit(h.nml, dif) * fcs.w + fcs.rgb) * fgi.w + fgi.rgb;
  return  h.srf.kd * lin;
}

// Function 193
float mapSeedNoLight(vec2 f)
{
    DecodeData(texelFetch( iChannel0, ivec2(f),0), seedCoord, seedColor);
    return length((floor(seedCoord)-floor(f)))-seedColor.z*circSizeMult*iResolution.x;
}

