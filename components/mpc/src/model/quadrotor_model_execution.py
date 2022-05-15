from ctypes import POINTER, cast, CDLL, c_void_p, c_char_p, c_double, c_int, c_int64, byref
import numpy as np
import time
import os


nx = 13
nu = 4
ny = nx + nu
ny_e = nx

model_name = "quadrotor_model"
lib_path = os.path.join(os.path.dirname(__file__), f"../../c_generated_code/libacados_ocp_solver_{model_name}.so")
shared_lib = CDLL(lib_path)

getattr(shared_lib, f"{model_name}_acados_create_capsule").restype = c_void_p
acados_create_capsule = getattr(shared_lib, f"{model_name}_acados_create_capsule")

getattr(shared_lib, f"{model_name}_acados_create").argtypes = [c_void_p]
getattr(shared_lib, f"{model_name}_acados_create").restype = c_int
acados_create = getattr(shared_lib, f"{model_name}_acados_create")

getattr(shared_lib, f"{model_name}_acados_get_nlp_opts").argtypes = [c_void_p]
getattr(shared_lib, f"{model_name}_acados_get_nlp_opts").restype = c_void_p
acados_get_nlp_opts = getattr(shared_lib, f"{model_name}_acados_get_nlp_opts")

getattr(shared_lib, f"{model_name}_acados_get_nlp_dims").argtypes = [c_void_p]
getattr(shared_lib, f"{model_name}_acados_get_nlp_dims").restype = c_void_p
acados_get_nlp_dims = getattr(shared_lib, f"{model_name}_acados_get_nlp_dims")

getattr(shared_lib, f"{model_name}_acados_get_nlp_config").argtypes = [c_void_p]
getattr(shared_lib, f"{model_name}_acados_get_nlp_config").restype = c_void_p
acados_get_nlp_config = getattr(shared_lib, f"{model_name}_acados_get_nlp_config")

getattr(shared_lib, f"{model_name}_acados_get_nlp_out").argtypes = [c_void_p]
getattr(shared_lib, f"{model_name}_acados_get_nlp_out").restype = c_void_p
acados_get_nlp_out = getattr(shared_lib, f"{model_name}_acados_get_nlp_out")

getattr(shared_lib, f"{model_name}_acados_get_sens_out").argtypes = [c_void_p]
getattr(shared_lib, f"{model_name}_acados_get_sens_out").restype = c_void_p
acados_get_sens_out = getattr(shared_lib, f"{model_name}_acados_get_sens_out")

getattr(shared_lib, f"{model_name}_acados_get_nlp_in").argtypes = [c_void_p]
getattr(shared_lib, f"{model_name}_acados_get_nlp_in").restype = c_void_p
acados_get_nlp_in = getattr(shared_lib, f"{model_name}_acados_get_nlp_in")

getattr(shared_lib, f"{model_name}_acados_get_nlp_solver").argtypes = [c_void_p]
getattr(shared_lib, f"{model_name}_acados_get_nlp_solver").restype = c_void_p
acados_get_nlp_solver = getattr(shared_lib, f"{model_name}_acados_get_nlp_solver")

getattr(shared_lib, f"{model_name}_acados_solve").argtypes = [c_void_p]
getattr(shared_lib, f"{model_name}_acados_solve").restype = c_int
acados_solve = getattr(shared_lib, f"{model_name}_acados_solve")


shared_lib.ocp_nlp_cost_model_set.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_char_p, c_void_p]
shared_lib.ocp_nlp_out_set.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_char_p, c_void_p]
shared_lib.ocp_nlp_constraints_model_set.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_char_p, c_void_p]
shared_lib.ocp_nlp_out_get.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_char_p, c_void_p]

def create_capsule():
    quadrotor_controller_capsule = acados_create_capsule()
    assert acados_create(quadrotor_controller_capsule)==0
    return {
        "capsule": quadrotor_controller_capsule,
        "nlp_opts": acados_get_nlp_opts(quadrotor_controller_capsule),
        "nlp_dims": acados_get_nlp_dims(quadrotor_controller_capsule),
        "nlp_config": acados_get_nlp_config(quadrotor_controller_capsule),
        "nlp_out": acados_get_nlp_out(quadrotor_controller_capsule),
        "sens_out": acados_get_sens_out(quadrotor_controller_capsule),
        "nlp_in": acados_get_nlp_in(quadrotor_controller_capsule),
        "nlp_solver": acados_get_nlp_solver(quadrotor_controller_capsule),
    }



# ocp_solver.shared_lib.ocp_nlp_cost_dims_get_from_attr.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_char_p, POINTER(c_int)]
# ocp_solver.shared_lib.ocp_nlp_cost_dims_get_from_attr.restype = c_int

# dims = np.ascontiguousarray(np.zeros((2,)), dtype=np.intc)
# dims_data = cast(dims.ctypes.data, POINTER(c_int))

# ocp_solver.shared_lib.ocp_nlp_cost_dims_get_from_attr(nlp_config, nlp_dims, nlp_out, stage, "yref".encode("utf-8"), dims_data)


def set_ocp(capsule, x0, traj, N, Tf):
    c = capsule
    x0 = np.ravel(x0.astype(float), order='F')
    x0_data = cast(x0.ctypes.data, POINTER(c_double))
    x0_data_p = cast((x0_data), c_void_p)

    for stage in range(N+1):
        target = traj[stage, :] if stage != N else traj[stage, :nx]
        target_data = cast(target.copy().ctypes.data, POINTER(c_double))
        target_data_p = cast((target_data), c_void_p)
        shared_lib.ocp_nlp_cost_model_set(c["nlp_config"], c["nlp_dims"], c["nlp_in"], c_int(stage), "yref".encode("utf-8"), target_data_p)

        initial_guess = traj[stage, :nx]
        initial_guess_data = cast(initial_guess.copy().ctypes.data, POINTER(c_double))
        initial_guess_data_p = cast((initial_guess_data), c_void_p)
        shared_lib.ocp_nlp_out_set(c["nlp_config"], c["nlp_dims"], c["nlp_out"], c_int(stage), "x".encode("utf-8"), initial_guess_data_p)

    shared_lib.ocp_nlp_constraints_model_set(c["nlp_config"], c["nlp_dims"], c["nlp_in"], c_int(0), "lbx".encode("utf-8"), x0_data_p)
    shared_lib.ocp_nlp_constraints_model_set(c["nlp_config"], c["nlp_dims"], c["nlp_in"], c_int(0), "ubx".encode("utf-8"), x0_data_p)

def get_ocp_result(capsule, N, Tf):
    c = capsule
    simX = np.ones((N+1, nx))
    simU = np.ones((N, nu))
    t = np.arange(N+1)/N * Tf

    def get(stage, field, dims):
        out = np.ascontiguousarray(np.zeros((dims,)), dtype=np.float64)
        out_data = cast(out.ctypes.data, POINTER(c_double))
        shared_lib.ocp_nlp_out_get(c["nlp_config"], c["nlp_dims"], c["nlp_out"], c_int(stage), field.encode("utf-8"), out_data)
        return out
    for i in range(N):
        simX[i,:] = get(i, "x", nx)
        simU[i,:] = get(i, "u", nu)

    simX[N,:] = get(N, "x", nx)
    return t, simX, simU

def control_trajectory(capsule, x0, traj, N, Tf):
    set_ocp(capsule, x0, traj, N, Tf)
    status = acados_solve(capsule["capsule"])
    return get_ocp_result(capsule, N, Tf)


