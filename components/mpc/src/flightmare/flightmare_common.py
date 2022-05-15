import os
import numpy as np
from scipy.spatial.transform.rotation import Rotation as R
from ruamel.yaml import YAML, dump, RoundTripDumper

from flightgym import QuadrotorEnv_v1
from rpg_baselines.envs import vec_env_wrapper as wrapper



def obs_to_state(obs, params):
    pt = obs[0, :3]
    # euler_zyx = obs[0, 3:3+3]
    # q_raw = R.from_euler("zyx", euler_zyx).as_quat()
    # qt = np.array([q_raw[3], *q_raw[:3]])
    # if qt[0] < 0:
    #     qt *= -1
    qt = obs[0, 3:3+4]
    vt = obs[0, 3+4:3+4+3]
    wt = obs[0, 3+4+3:]

    p = params["fromTargetRotationMatrix"] @ pt
    orientationTarget = np.quaternion(*qt)
    q = params["fromTargetRotation"] * orientationTarget * params["toTargetRotation"]
    v = params["fromTargetRotationMatrix"] @ vt
    w = params["fromTargetRotationMatrix"] @ wt
    return np.concatenate([p, np.array([q.w, q.x, q.y, q.z]), v, w])


def create_env():
    cfg = YAML().load(open(os.path.join(os.path.dirname(__file__), "config/vec_env.yaml")))
    cfg["env"]["num_envs"] = 1
    cfg["env"]["num_threads"] = 1
    cfg["env"]["render"] = "yes"
    env = wrapper.FlightEnvVec(QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False))
    return env