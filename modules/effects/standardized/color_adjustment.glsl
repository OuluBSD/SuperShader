// Color adjustment module
// Standardized color adjustment functions

// Adjust brightness, contrast and saturation
vec3 AdjustColor(vec3 color, float brightness, float contrast, float saturation) {
    // Brightness
    color += brightness;
    
    // Contrast
    color = (color - 0.5) * contrast + 0.5;
    
    // Saturation
    vec3 grey = vec3(dot(color, vec3(0.299, 0.587, 0.114)));
    color = mix(grey, color, saturation);
    
    return clamp(color, 0.0, 1.0);
}

// Apply gamma correction
vec3 ApplyGamma(vec3 color, float gamma) {
    return pow(color, vec3(1.0 / gamma));
}

// Adjust hue, saturation and lightness
vec3 HSLAdjust(vec3 color, float hueAdjust, float satAdjust, float lightAdjust) {
    // Convert RGB to HSL
    vec3 hsl = RGBToHSL(color);
    
    // Adjust HSL values
    hsl.x += hueAdjust;
    hsl.y = clamp(hsl.y * satAdjust, 0.0, 1.0);
    hsl.z = clamp(hsl.z + lightAdjust, 0.0, 1.0);
    
    // Convert back to RGB
    return HSLToRGB(hsl);
}

// Convert RGB to HSL
vec3 RGBToHSL(vec3 color) {
    float minVal = min(min(color.r, color.g), color.b);
    float maxVal = max(max(color.r, color.g), color.b);
    float delta = maxVal - minVal;
    
    vec3 hsl = vec3(0.0);
    hsl.z = (maxVal + minVal) / 2.0;
    
    if(delta == 0.0) {
        hsl.x = 0.0;
        hsl.y = 0.0;
    } else {
        if(hsl.z < 0.5)
            hsl.y = delta / (maxVal + minVal);
        else
            hsl.y = delta / (2.0 - maxVal - minVal);
        
        if(color.r == maxVal)
            hsl.x = (color.g - color.b) / delta;
        else if(color.g == maxVal)
            hsl.x = 2.0 + (color.b - color.r) / delta;
        else
            hsl.x = 4.0 + (color.r - color.g) / delta;
        
        hsl.x = hsl.x / 6.0;
        if(hsl.x < 0.0)
            hsl.x += 1.0;
    }
    
    return hsl;
}

// Convert HSL to RGB
vec3 HSLToRGB(vec3 hsl) {
    vec3 rgb;
    
    if(hsl.y == 0.0) {
        rgb = vec3(hsl.z);
    } else {
        float q = hsl.z < 0.5 ? 
            hsl.z * (1.0 + hsl.y) : 
            hsl.z + hsl.y - hsl.z * hsl.y;
        float p = 2.0 * hsl.z - q;
        
        rgb.r = HueToRGB(p, q, hsl.x + (1.0/3.0));
        rgb.g = HueToRGB(p, q, hsl.x);
        rgb.b = HueToRGB(p, q, hsl.x - (1.0/3.0));
    }
    
    return rgb;
}

// Helper for HSL to RGB conversion
float HueToRGB(float p, float q, float t) {
    if(t < 0.0) t += 1.0;
    if(t > 1.0) t -= 1.0;
    if(t < 1.0/6.0) return p + (q - p) * 6.0 * t;
    if(t < 1.0/2.0) return q;
    if(t < 2.0/3.0) return p + (q - p) * (2.0/3.0 - t) * 6.0;
    return p;
}
