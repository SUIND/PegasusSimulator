# Camera configs

import numpy as np

width, height = 640, 400
pixel_size = 3  # microns
f_stop = 2.8
focus_distance = 1000000

# -----------------------------
# Camera intrinsics and distortion
# -----------------------------
intrinsics = {
    "top_left":  ([453.4450325149651, 451.90515125268513, 334.42633913038867, 201.81057455311947],
                  [0.046338176370833116, -0.06177658870039574, -0.00011460275090373664, -0.000954180108467503]),
    "top_right": ([452.57768991422444, 451.0119769604357, 306.58177884031005, 209.79592882031682],
                  [0.04253023174183001, -0.05451701933236324, -0.0005814397626365427, -0.0009656429606147007]),
    "bottom_left": ([454.5326666380746, 452.99169214576324, 346.70861607351327, 209.68146239129342],
                    [0.0430742610695452, -0.05338680726177966, -0.0011106341725500438, 0.0001868962257358484]),
    "bottom_right": ([455.08694881321827, 453.5822333643735, 307.9468133450323, 202.38959247954645],
                     [0.043520952114209414, -0.05596988344093367, -0.0003596477838297661, -0.00012052365632834432])
}

# -----------------------------
# Relative transforms 
# -----------------------------
T_top = np.array([
    [ 0.9999656338561949, -0.003953487738938295,  0.007287046127152642, -0.1998715819981644],
    [ 0.0039227589849132735,  0.9999833746929159,  0.004226381395053844,  0.0021412607430401527],
    [-0.007303633924798527, -0.0041976508249543165, 0.9999645177000167,  0.0018455186766273962],
    [0.0,                   0.0,                   0.0,                    1.0]
])

T_bottom = np.array([
    [ 0.9999350402657172,  0.010305926759755939, -0.004868585258750211, -0.16045327333884726],
    [-0.010327093435662998, 0.9999372442748176,  -0.004342655092883615,  0.002353587196452448],
    [ 0.004823524641821544, 0.004392651330029303, 0.9999787188857188,   0.0001303452394150501],
    [0.0,                   0.0,                   0.0,                 1.0]
])

fx, fy, cx, cy = intrinsics["top_left"][0]
dist = intrinsics["top_left"][1]

TOP_LEFT_CFG = { 
    "position": np.array([0.15, 0.10, 0.0]),
    "resolution": (width, height),
    "frequency": 15,
    "intrinsics": np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]),
    "distortion_coefficients": dist,

}

fx, fy, cx, cy = intrinsics["top_right"][0]
dist = intrinsics["top_right"][1]

TOP_RIGHT_CFG = { 
    "position": np.array([0.15, -0.10, 0.0]),
    "resolution": (width, height),
    "frequency": 15,
    "intrinsics": np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]),
    "distortion_coefficients": dist,

}

fx, fy, cx, cy = intrinsics["bottom_left"][0]
dist = intrinsics["bottom_left"][1]

BOTTOM_LEFT_CFG = { 
    "position": np.array([0.15, 0.08, 0.0]),
    "orientation": np.array([0, -15, 180]),
    "resolution": (width, height),
    "frequency": 15,
    "intrinsics": np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]),
    "distortion_coefficients": dist,

}

fx, fy, cx, cy = intrinsics["bottom_right"][0]
dist = intrinsics["bottom_right"][1]

BOTTOM_RIGHT_CFG = { 
    "position": np.array([0.15, -0.08, 0.0]),
    "orientation": np.array([0, -15, 180]),
    "resolution": (width, height),
    "frequency": 15,
    "intrinsics": np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]),
    "distortion_coefficients": dist,

}