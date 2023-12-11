import numpy as np
import matplotlib.pyplot as plt

pi = np.pi


class IncomingRadiation:

    def __init__(self, n_wb):
        self.n_wb = n_wb
        self.direct = None
        self.diffuse = None
        self.sza = None

    # should be flux over horizontal surface
    def add_direct(self, direct, sza):
        direct = np.array(direct)

        assert len(direct) == self.n_wb, "Ensure proper number of wavebands"

        for i in direct:
            assert i >= 0, "Incoming direct radiation should be non-negative"

        assert 0 <= sza <= pi/2, "Solar zenith angle must be above the horizon"

        self.direct = direct
        self.sza = sza

    # assume isotropic diffuse
    # input should be flux over horizontal surface
    # I = F/pi
    def add_diffuse(self, diffuse):
        diffuse = np.array(diffuse)

        assert len(diffuse) == self.n_wb, "Ensure proper number of wavebands"

        for i in diffuse:
            assert i >= 0, "Incoming diffuse radiation should be non-negative"

        self.diffuse = diffuse





