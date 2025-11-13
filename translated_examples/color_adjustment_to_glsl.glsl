vec3 adjust_brightness_contrast(vec3 color, float brightness, float contrast) {
    vec3 adjusted; = color + brightness;
    adjusted = (adjusted - 0.5) * contrast + 0.5;
    return adjusted;
}