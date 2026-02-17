import numpy as np

def TwistTanh(pos):
    B0 = 1.0
    By0 = 1
    Bz0 = 1.0
    L = 1
    return np.array([0, By0 , Bz0* np.tanh(pos[1]/L)])


def B_model(pos):
    return TwistTanh(pos)
