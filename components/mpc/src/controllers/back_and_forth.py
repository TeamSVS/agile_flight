import numpy as np
import time

pos_target_source_initial = None

def setpoint(x):
    global pos_target_source_initial

    pos_source = x[:3]
    if pos_target_source_initial is None:
        pos_target_source_initial = pos_source.copy()
    pos_target_source = pos_target_source_initial.copy()
    # pos_target_source[2] = -5

    phi = 45/180*np.pi
    if pos_source[2] < -3:
        if np.floor(time.time() % 6) > 3:
            phi = 0/180*np.pi
            pos_target_source[1] = 1
        else:
            phi = 90/180*np.pi
            pos_target_source[1] = -1

    pos_target_source_diff = pos_target_source - pos_source
    pos_limit = 1
    if np.linalg.norm(pos_target_source_diff) > pos_limit:
        pos_target_source = pos_source + pos_target_source_diff/np.linalg.norm(pos_target_source_diff) * pos_limit


    setpoint = np.array([
        *pos_target_source,
        # 1, 0, 0, 0,
        np.cos(phi/2), *(np.sin(phi/2)*np.array([0, 0, 1])),
        0, 0, 0,
        0, 0, 0,
    ]).astype(float)
    return setpoint