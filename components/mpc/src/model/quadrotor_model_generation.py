from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, cross, Function
import casadi
import numpy as np
import functools
import time


def quaternion_derivative(q, omega):
    '''
    using LinearAlgebra
    using Symbolics
    @variables qw qx qy qz wx wy wz
    w = Quaternion(0, wx, wy, wz)
    q = Quaternion(qw, qx, qy, qz)
    qd = 1/2 * q * w
    out = [
        qd.s,
        qd.v1,
        qd.v2,
        qd.v3,
    ]
    '''
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    wx, wy, wz = omega[0], omega[1], omega[2]
    return vertcat(
        -0.5*qx*wx - 0.5*qy*wy - 0.5*qz*wz,
         0.5*qw*wx + 0.5*qy*wz - 0.5*qz*wy,
         0.5*qw*wy + 0.5*qz*wx - 0.5*qx*wz,
         0.5*qw*wz + 0.5*qx*wy - 0.5*qy*wx,
    )


def rotate_vector_by_quaternion(q, v):
    '''
    using LinearAlgebra
    using Symbolics
    @variables qw qx qy qz vx vy vz
    q = Quaternion(qw, qx, qy, qz)
    v = [vx, vy, vz]
    vout = q * Quaternion(0, v...) * conj(q)
    out = [
        vout.v1,
        vout.v2,
        vout.v3,
    ]
    '''
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    vx, vy, vz = v[0], v[1], v[2]
    return vertcat(
        qw*(qw*vx + qy*vz - qz*vy) + qy*(qx*vy + qw*vz - qy*vx) - qz*(qw*vy + qz*vx - qx*vz) - qx*(-qx*vx - qy*vy - qz*vz),
        qw*(qw*vy + qz*vx - qx*vz) + qz*(qw*vx + qy*vz - qz*vy) - qx*(qx*vy + qw*vz - qy*vx) - qy*(-qx*vx - qy*vy - qz*vz),
        qw*(qx*vy + qw*vz - qy*vx) + qx*(qw*vy + qz*vx - qx*vz) - qy*(qw*vx + qy*vz - qz*vy) - qz*(-qx*vx - qy*vy - qz*vz),
    )


def rotor_thrusts(a, params):
    if params["control_level"] == "rpm":
        thrusts = SX(*[params["motor_thrust_constants"].dot([a[i] ** 2, a[i], 1]) for i in enumerate(a)])
    elif params["control_level"] == "thrust":
        thrusts = a
    else:
        raise Exception("Control level not implemented")
    return thrusts

def propeller_thrust_model(a, params):
    thrusts = rotor_thrusts(a, params)
    return [SX(direction * thrusts[i]) for i, direction in enumerate(params["rotor_thrust_directions"])]

def propeller_torque_model(a, params):
    thrusts = rotor_thrusts(a, params)
    return [SX((direction * thrusts[i] * params["motor_torque_constant"])) for i, direction in enumerate(params["rotor_torque_directions"])]

def combined_thrust(f):
    return functools.reduce(lambda a,c: a+c, f)

def combined_torque(t, f, params):
    acc = np.zeros(3)
    for t, f, r in zip(t, f, params["rotor_positions"]):
        acc += t + cross(r, f)
    return acc

def quadrotor_linear_dynamics(q, f_prop, params):
    return 1/params["mass"]*rotate_vector_by_quaternion(q, f_prop) + np.array([0, 0, params["g"]])

def quadrotor_angular_dynamics(w, t_prop, params):
    J_inv = np.linalg.inv(params["J"])
    return J_inv @ (t_prop - cross(w, params["J"] @ w))



def quadrotor_model(params):
    model_name = 'quadrotor_model'

    # constants

    # states
    x = SX.sym('x', 3)
    q = SX.sym('q', 4)
    v = SX.sym('v', 3)
    w = SX.sym('w', 3)

    s = vertcat(x, q, v, w)
    a = SX.sym("rpm", 4)

    propeller_thrusts = propeller_thrust_model(a, params)


    f_expl = vertcat(
        v,
        quaternion_derivative(q, w),
        quadrotor_linear_dynamics(q, combined_thrust(propeller_thrusts), params),
        quadrotor_angular_dynamics(w, combined_torque(propeller_torque_model(a, params), propeller_thrusts, params), params),
    )

    x_dot = SX.sym('x_dot', 3, 1)
    q_dot = SX.sym('q_dot', 4, 1)
    v_dot = SX.sym('v_dot', 3, 1)
    w_dot = SX.sym('w_dot', 3, 1)

    s_dot = vertcat(x_dot, q_dot, v_dot, w_dot)

    f_impl = s_dot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = s
    model.xdot = s_dot
    model.u = a
    model.name = model_name

    return model, Function("dsdt", [s, a], [f_expl])



def generate(params):
    from acados_template import AcadosOcp, AcadosOcpSolver

    ocp = AcadosOcp()

    model, dsdt = quadrotor_model(params)
    ocp.model = model

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    ny_e = nx


    ocp.dims.N = params["N"]


    ocp.cost.W = np.diag([*params["Q"], *params["R"]])
    ocp.cost.W_e = np.diag(params["Qe"])

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"


    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :] = np.eye(nx)
    ocp.cost.Vx_e = np.eye(nx)


    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[nx:, :] = np.eye(nu)

    yref = np.array([
        0, 0, 0,
        1, 0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0, 0
    ])
    ocp.cost.yref = yref

    yref_e = yref[:13]
    ocp.cost.yref_e = yref_e

    ocp.constraints.lbu = np.ones(4) * params["min_output"]
    ocp.constraints.ubu = np.ones(4) * params["max_output"]
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # ocp.constraints.ubx = np.ones(3)*0.1
    # ocp.constraints.lbx = -np.ones(3) * 100
    # ocp.constraints.idxbx = np.array([0, 1, 2])


    x0 = np.array([
        0, 0, -1,
        # 0, 0, 0,
        0.9961946980917455, 0.08715574274765817, 0, 0,
        # 1, 0, 0, 0,
        # 0, 1, 0, 0,
        # 0.08715574274765817, 0.9961946980917455, 0, 0,
        0, 0, 0,
        0, 0, 0,
    ])

    ocp.constraints.x0 = x0

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    # ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tf = params["Tf"]
    ocp.solver_options.nlp_solver_max_iter = params["nlp_solver_max_iter"]


    ocp_solver = AcadosOcpSolver(ocp)

def test_generated(params):
    from ..model_test.quadrotor_model_test import test, test_speed
    # test()
    test_speed(params)

if __name__ == "__main__":
    from src.params import flightmare
    params = flightmare.params()
    generate(params)
    test_generated(params)

