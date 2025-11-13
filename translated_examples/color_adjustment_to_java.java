public class ShaderFunctions {

void adjust_brightness_contrast(Vector3f color, float brightness, float contrast) {
    Vector3f adjusted; = color + brightness;
    adjusted = (adjusted - 0.5) * contrast + 0.5;
    return adjusted;
}
}