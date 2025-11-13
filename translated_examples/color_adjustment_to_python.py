import numpy as np

def adjust_brightness_contrast(np.ndarray color, float brightness, float contrast):
    adjusted = None  # : vec3 = color + brightness;
    adjusted = (adjusted - 0.5) * contrast + 0.5;
    return adjusted
}