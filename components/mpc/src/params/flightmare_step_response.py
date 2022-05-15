from .flightmare import params as params_stub


def params():
    p = params_stub()
    p["Tf"] = 1
    p["N"] = int(round(p["Tf"] / 0.02))
    return p