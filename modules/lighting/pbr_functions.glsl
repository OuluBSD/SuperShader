// Reusable Pbr Lighting Functions
// Automatically extracted from lighting-related shaders

// Function 1
vec3 PBR_HDRCubemap(vec3 sampleDir, float LOD_01)
{
    vec3 linearGammaColor_sharp = PBR_HDRremap(pow(texture( iChannel2, sampleDir ).rgb,vec3(2.2)));
    vec3 linearGammaColor_blur  = PBR_HDRremap(pow(texture( iChannel3, sampleDir ).rgb,vec3(1)));
    vec3 linearGammaColor = mix(linearGammaColor_sharp,linearGammaColor_blur,saturate(LOD_01));
    return linearGammaColor;
}

// Function 2
vec3 PBR_visitSamples(vec3 V, vec3 N, float roughness, bool metallic, vec3 ior_n, vec3 ior_k )
{
    //Direct relection vector
    vec3 vCenter = reflect(-V,N);
    
    //------------------------------------------------
	//  Randomized Samples : more realistic, but
    //  a lot of samples before it stabilizes 
    //------------------------------------------------
    float randomness_range = 0.75; //Cover only the closest 75% of the distribution. Reduces range, but improves stability.
    float fIdx = 0.0;              //valid range = [0.5-1.0]. Note : it is physically correct at 1.0.
    const int ITER_RDM = 05;
    const float w_rdm = 1.0/float(ITER_RDM);
    vec3 totalRandom = vec3(0.0);
    for(int i=0; i < ITER_RDM; ++i)
    {
        //Random jitter note : very sensitive to hash quality (patterns & artifacts).
        vec2 jitter = hash22(fIdx*10.0+vCenter.xy*100.0);
    	float angularRange = 0.;    
        vec3 sampleDir    = PBR_nudgeSample(vCenter, roughness, jitter.x*randomness_range, jitter.y, angularRange);
        vec3 sampleColor  = PBR_HDRCubemap( sampleDir, angularRange/MIPMAP_SWITCH);
        vec3 contribution = PBR_Equation(V, sampleDir, N, roughness, ior_n, ior_k, metallic, true)*w_rdm;
    	totalRandom += contribution*sampleColor;
		++fIdx;
    }
    
    //------------------------------------------------
	//  Fixed Samples : More stable, but creates
    //  sampling pattern artifacts and the reach is
    //  limited.
    //------------------------------------------------
    fIdx = 0.0;
    const int ITER_FIXED = 15;
    const float w_fixed = 1.0/float(ITER_FIXED); //Sample
    vec3 totalFixed = vec3(0.0);
    for(int i=0; i < ITER_FIXED; ++i)
    {
        //Stable pseudo-random jitter (to improve stability with low sample count)
        //Beware here! second component controls the sampling pattern "swirl", and it must be choosen 
        //             so that samples do not align by doing complete 360deg cycles at each iteration.
        vec2 jitter = vec2( clamp(w_fixed*fIdx,0.0,0.50),
                            fract(w_fixed*fIdx*1.25)+3.14*fIdx);
        float angularRange = 0.;
        vec3 sampleDir    = PBR_nudgeSample(vCenter, roughness, jitter.x, jitter.y, angularRange);
        vec3 sampleColor  = PBR_HDRCubemap( sampleDir, angularRange/MIPMAP_SWITCH);
        vec3 contribution = PBR_Equation(V, sampleDir, N, roughness, ior_n, ior_k, metallic, true)*w_fixed;
        totalFixed += contribution*sampleColor;
		++fIdx;
    }
    
    return (totalRandom*float(ITER_RDM)+totalFixed*float(ITER_FIXED))/(float(ITER_RDM)+float(ITER_FIXED));
}

// Function 3
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

// Function 4
vec3 PBR_HDRremap(vec3 c)
{
    float fHDR = smoothstep(2.900,3.0,c.x+c.y+c.z);
    //vec3 cRedSky   = mix(c,1.3*vec3(4.5,2.5,2.0),fHDR);
    vec3 cBlueSky  = mix(c,1.8*vec3(2.0,2.5,3.0),fHDR);
    return cBlueSky;//mix(cRedSky,cBlueSky,SKY_COLOR);
}

// Function 5
vec3 ShadePbr(vec3 p,vec3 rd,vec3 ro,float matId)
{    
    // Check matId ???
    
    // Material
    Material mat = materials[int(matId)];
    mat.Roughness = max(mat.Roughness,0.005);
    
    
    vec3 F0 = vec3(0.04);					// Base normal incidence for 
    										// non-conductors (average of some materials)
    F0 = mix(F0,mat.Albedo,mat.Metalness);	// If it is a metal, take the normal incidence (color)
    										// from the albedo as metals should not have albedo color
    
    // Parameters
    vec3 eye = normalize(ro - p);
    vec3 n = SceneNormal(p);
    vec3 r = reflect(-eye,n);
    float ndv = max(dot(n,eye),0.0);
    
    float sAcum = float(kLights);
    vec3 acum = vec3(0.0);
    for(int i = 0; i < kLights; i++)
    {
        // Per-light parameters
        vec3 lp = lights[i].Position;
   	 	vec3 ld = normalize(lp - p);
        vec3 h = normalize(ld + eye);
    	float ndl = max(dot(n,ld),0.0);
        float ndh = max(dot(n,h),0.0);
        
        // Diffuse
        vec3 diffuseBRDF = mat.Albedo / kPi;

        // Specular
        float D = Distribution(ndh,mat.Roughness);
        float G = Geometry(ndv,ndl,mat.Roughness);
        vec3 F = Fresnel(ndv,F0);
        
        vec3 specularBRDFNom = D * G * F;
        float specularBRDFDenom = 4.0 * max(ndv * ndl, 0.0) + 0.001; 	// add bias to prevent
        														 		// division by 0
        vec3 specularBRDF = specularBRDFNom / specularBRDFDenom;
        
        // Outgoing light can't exced 1
        vec3 kS = F;
        vec3 kD = 1.0 - kS;
        kD *= 1.0 - mat.Metalness;
            
        vec3 finalCol = (kD * diffuseBRDF + specularBRDF);
        finalCol = finalCol * ndl * lights[i].Color;
        acum += finalCol;
        
        // Shadow
        vec3 sDir = normalize(lp - p);
        sAcum -= Shadow(p,sDir);
    }
    
    // IBL
   	vec3 F = FresnelRoughness(ndv,F0,mat.Roughness);
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
   	kD *= 1.0 - mat.Metalness;
    vec3 totalIBL = vec3(0.0);
    
    // Diffuse IBL
    vec3 diffuseIBL = IrradianceMap(n) * mat.Albedo * kD * 1.0;
    
    // Specular IBL
    vec3 specularIBL = ReflectanceMap(r,mat.Roughness,n) * F;
    
    totalIBL = kD * diffuseIBL + specularIBL;
    
    // Shadowing
    float finalShadow = max(sAcum / float(kLights),0.02);
    
    return (acum + totalIBL) * max(finalShadow,0.15);
}

// Function 6
vec3 PBR_nudgeSample(vec3 sampleDir, float roughness, float e1, float e2, out float range)
{
    const float PI = 3.14159;
    //Importance sampling :
    //Source : http://www.codinglabs.net/article_physically_based_rendering_cook_torrance.aspx
    //The higher the roughness, the broader the range.
    //In any case, wide angles are less probable than narrow angles.
    range = atan( roughness*sqrt(e1)/sqrt(1.0-e1) );
    //Circular angle has an even distribution (could be improved?).
	float phi = 2.0*PI*e2;
    
	vec3 up = vec3(0,1,0); //arbitrary
	vec3 tAxis = cross(up,sampleDir);
	mat3 m1 = UTIL_axisRotationMatrix(normalize(tAxis),range);
	mat3 m2 = UTIL_axisRotationMatrix(normalize(sampleDir), phi);
        
	return sampleDir*m1*m2;
}

// Function 7
vec3 PBR_Equation(vec3 V, vec3 L, vec3 N, float roughness, vec3 ior_n, vec3 ior_k, const bool metallic, const bool bIBL)
{
    float cosT = saturate( dot(L, N) );
    float sinT = sqrt( 1.0 - cosT * cosT);
    
	vec3 H = normalize(L+V);
	float NdotH = dot(N,H);//Nn.H;
	float NdotL = dot(N,L);//Nn.Ln;
	float VdotH = dot(V,H);//Vn.H;
    float NdotV = dot(N,V);//Nn.Vn;
    
    //-----------------------------------------
	//            Distribution Term
    //-----------------------------------------
    float PI = 3.14159;
    float alpha2 = roughness * roughness;
    float NoH2 = NdotH * NdotH;
    float den = NoH2*(alpha2-1.0)+1.0;
    float D = 1.0; //Distribution term is externalized from IBL version
    if(!bIBL)
        D = (NdotH>0.)?alpha2/(PI*den*den):0.0; //GGX Distribution.
	
    //-----------------------------------------
	//            Fresnel Term
    //-----------------------------------------
    vec3 F;
    if(metallic)
    {
        //Source: http://sirkan.iit.bme.hu/~szirmay/fresnel.pdf p.3 above fig 5
        float cos_theta = 1.0-NdotV;//REVIEWME : NdotV or NdotL ?
        F =  ((ior_n-1.)*(ior_n-1.)+ior_k*ior_k+4.*ior_n*pow(1.-cos_theta,5.))
		    /((ior_n+1.)*(ior_n+1.)+ior_k*ior_k);
    }
    else
    {
        //Fresnel Schlick Dielectric formula 
        //Sources: https://en.wikipedia.org/wiki/Schlick%27s_approximation
        //          http://www.codinglabs.net/article_physically_based_rendering_cook_torrance.aspx
        //Note: R/G/B do not really differ for dielectric materials
        float F0 = abs ((1.0 - ior_n.x) / (1.0 + ior_n.x));
  		F = vec3(F0 + (1.-F0) * pow( 1. - VdotH, 5.));
    }
    
    //-----------------------------------------
	//            Geometric term
    //-----------------------------------------
    //Source: Real Shading in Unreal Engine 4 2013 Siggraph Presentation
    //https://de45xmedrsdbp.cloudfront.net/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf p.3/59
    //k = Schlick model (IBL) : Disney's modification to reduce hotness (point light)
    float k = bIBL?(roughness*roughness/2.0):(roughness+1.)*(roughness+1.)/8.; 
    float Gl = max(NdotL,0.)/(NdotL*(1.0-k)+k);
    float Gv = max(NdotV,0.)/(NdotV*(1.0-k)+k);
    float G = Gl*Gv;
    
    //-----------------------------------------
	//     PBR Equation (ABL & IBL versions)
    //-----------------------------------------
    //Two flavors of the PBR equation (IBL/point light).
    //Personal addition: This parameter softens up the transition at grazing angles (otherwise too sharp IMHO).
    float softTr = 0.1; // Valid range : [0.001-0.25]. It will reduce reflexivity on edges when too high, however.
    //Personal addition: This parameter limits the reflexivity loss at 90deg viewing angle (black spot in the middle?).
    float angleLim = 0.15; // Valid range : [0-0.75] (Above 1.0, become very mirror-like and diverges from a physically plausible result)
    //Source: http://www.codinglabs.net/article_physically_based_rendering_cook_torrance.aspx
    if(bIBL)
        return (F*G*(angleLim+sinT)/(angleLim+1.0) / (4.*NdotV*saturate(NdotH)*(1.0-softTr)+softTr)); //IBL
    else
        return D*F*G / (4.*NdotV*NdotL*(1.0-softTr)+softTr);	//ABL
}

// Function 8
vec3 shadePBR (in vec3 ro, in vec3 rd, in float d, in int id)
{
    vec3 p = ro + d * rd;
    vec3 nor = normal (p, d*EPSILON);

    // "material" hard-coded for the moment
    vec3 albedo =     (id == 1) ? vec3(.7,.55,.45) : (id == 2) ? vec3 (.9) : (id == 3) ? vec3 (.2, .4, .9) : vec3 (.9, .4, .1);
    float metallic =  (id == 1) ? .9 : (id == 2) ? 0.0 : (id == 3) ? .0 : .0; 
    float roughness = (id == 1) ? .125 : (id == 2) ? .5 : (id == 3) ? .5 : .2;
    float ao = 1.;

    // lights hard-coded as well atm
    vec3 lightColors[4];
    lightColors[0] = vec3 (.9, .9, .9) * 120.;
    lightColors[1] = vec3 (.9, .25, .9) * 275.;
    lightColors[2] = vec3 (.25, .9, .9) * 275.;
    lightColors[3] = vec3 (.25, .9, .25) * 275.;

    vec3 lightPositions[4];
    lightPositions[0] = vec3 (.0, .0, .0);
    lightPositions[1] = vec3 (-1.1, 1.5, -2.);
    lightPositions[2] = vec3 (-1., .75, -2.2);
    lightPositions[3] = vec3 (.0, 2.75, -2.2);

	vec3 N = normalize (nor);
    vec3 V = normalize (ro - p);

    vec3 F0 = vec3 (0.04); 
    F0 = mix (F0, albedo, metallic);
    vec3 kD = vec3(.0);
	           
    // reflectance equation - cutting down on the light boost preformance of course
    vec3 Lo = vec3 (.0);
    for(int i = 0; i < 4; ++i) 
    {
        // calculate per-light radiance
        vec3 L = normalize(lightPositions[i] - p);
        vec3 H = normalize(V + L);
        float distance    = length(lightPositions[i] - p);
        float attenuation = 2.5 / (distance * distance);
        vec3 radiance     = lightColors[i] * attenuation;        
        
        // cook-torrance brdf
        float aDirect = .125 * pow (roughness + 1., 2.);
        float aIBL = .5 * roughness * roughness;
        float NDF = DistributionGGX(N, H, roughness);        
        float G   = GeometrySmith(N, V, L, roughness);      
        vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0, roughness);       
        
        vec3 kS = F;
        kD = vec3(1.) - kS;
        kD *= 1. - metallic;	  
        
        vec3 nominator    = NDF * G * F;
        float denominator = 4. * max(dot(N, V), 0.0) * max(dot(N, L), 0.0);
        vec3 specular     = nominator / max(denominator, .001);  

        // add to outgoing radiance Lo
        float NdotL = max(dot(N, L), 0.0);                
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;

        if (DO_SHADOWS) {
	    	Lo *= shadow (p, L);
        }
    }

    vec3 irradiance = texture (iChannel0, N).rgb;
    vec3 diffuse    = irradiance * albedo;
    vec3 ambient    = (kD * diffuse) * ao;

    return ambient + Lo;
}

