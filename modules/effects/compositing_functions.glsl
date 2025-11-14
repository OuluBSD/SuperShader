// Reusable Compositing Effect Functions
// Automatically extracted from effect-related shaders

// Function 1
bool is_demo_stage_composite()
{
    return is_demo_stage_composite(g_demo_stage);
}

// Function 2
vec4 CompositeSample(vec2 UV) {
	vec2 InverseRes = 1.0 / iResolution.xy;
	vec2 InverseP = vec2(P, 0.0) * InverseRes;
	
	// UVs for four linearly-interpolated samples spaced 0.25 texels apart
	vec2 C0 = UV;
	vec2 C1 = UV + InverseP * 0.25;
	vec2 C2 = UV + InverseP * 0.50;
	vec2 C3 = UV + InverseP * 0.75;
	vec4 Cx = vec4(C0.x, C1.x, C2.x, C3.x);
	vec4 Cy = vec4(C0.y, C1.y, C2.y, C3.y);

	vec3 Texel0 = texture(iChannel0, C0).rgb;
	vec3 Texel1 = texture(iChannel0, C1).rgb;
	vec3 Texel2 = texture(iChannel0, C2).rgb;
	vec3 Texel3 = texture(iChannel0, C3).rgb;
	
	// Calculated the expected time of the sample.
	vec4 T = A * Cy * vec4(iResolution.x) * Two + B + Cx;

	const vec3 YTransform = vec3(0.299, 0.587, 0.114);
	const vec3 ITransform = vec3(0.595716, -0.274453, -0.321263);
	const vec3 QTransform = vec3(0.211456, -0.522591, 0.311135);

	float Y0 = dot(Texel0, YTransform);
	float Y1 = dot(Texel1, YTransform);
	float Y2 = dot(Texel2, YTransform);
	float Y3 = dot(Texel3, YTransform);
	vec4 Y = vec4(Y0, Y1, Y2, Y3);

	float I0 = dot(Texel0, ITransform);
	float I1 = dot(Texel1, ITransform);
	float I2 = dot(Texel2, ITransform);
	float I3 = dot(Texel3, ITransform);
	vec4 I = vec4(I0, I1, I2, I3);

	float Q0 = dot(Texel0, QTransform);
	float Q1 = dot(Texel1, QTransform);
	float Q2 = dot(Texel2, QTransform);
	float Q3 = dot(Texel3, QTransform);
	vec4 Q = vec4(Q0, Q1, Q2, Q3);

	vec4 W = vec4(Pi2 * CCFrequency * ScanTime);
	vec4 Encoded = Y + I * cos(T * W) + Q * sin(T * W);
	return (Encoded - MinC) / CRange;
}

// Function 3
vec4 CompositeSample(vec2 UV, vec2 InverseRes) {
	vec2 InverseP = vec2(P, 0.0) * InverseRes;
	
	// UVs for four linearly-interpolated samples spaced 0.25 texels apart
	vec2 C0 = UV;
	vec2 C1 = UV + InverseP * 0.25;
	vec2 C2 = UV + InverseP * 0.50;
	vec2 C3 = UV + InverseP * 0.75;
	vec4 Cx = vec4(C0.x, C1.x, C2.x, C3.x);
	vec4 Cy = vec4(C0.y, C1.y, C2.y, C3.y);

	vec4 Texel0 = texture(iChannel0, C0);
	vec4 Texel1 = texture(iChannel0, C1);
	vec4 Texel2 = texture(iChannel0, C2);
	vec4 Texel3 = texture(iChannel0, C3);
	
	float Frequency = CCFrequency;
	//Frequency = Frequency;// Uncomment for bad color sync + (sin(UV.y * 2.0 - 1.0) / CCFrequency) * 0.001;

	// Calculated the expected time of the sample.
	vec4 T = A2 * Cy * vec4(iChannelResolution[0].y) + B + Cx;
	vec4 W = vec4(Pi2ScanTime * Frequency);
	vec4 TW = T * W;
	vec4 Y = vec4(dot(Texel0, YTransform), dot(Texel1, YTransform), dot(Texel2, YTransform), dot(Texel3, YTransform));
	vec4 I = vec4(dot(Texel0, ITransform), dot(Texel1, ITransform), dot(Texel2, ITransform), dot(Texel3, ITransform));
	vec4 Q = vec4(dot(Texel0, QTransform), dot(Texel1, QTransform), dot(Texel2, QTransform), dot(Texel3, QTransform));

	vec4 Encoded = Y + I * cos(TW) + Q * sin(TW);
	return (Encoded - MinC) * InvCRange;
}

// Function 4
bool is_demo_stage_composite(int stage)
{
    return uint(stage - DEMO_STAGE_DEPTH) >= uint(DEMO_STAGE_COMPOSITE - DEMO_STAGE_DEPTH);
}

// Function 5
void CompositeLayers(vec3 R0, vec3 T0, vec3 R1, vec3 T1, out vec3 R, out vec3 T) {
    vec3 tmp = vec3(1.0) / (vec3(1.0) - R0 * R1);
    R = R0 + T0 * T0 * R1 * tmp;
    T = T0 * T1 * tmp;
}

