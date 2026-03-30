import numpy as np

def bb_file(PATH):

    bethe_data = np.loadtxt(PATH)
    resrange = bethe_data[:, 0]
    bb = bethe_data[:, 1]
    res = np.flip(resrange).copy()

    return bb, res
