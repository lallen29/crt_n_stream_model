import DefaultOrientationFunctions as Orient
import DefaultPhaseFunctions as Phase
import numpy as np
import scipy.integrate as integrate

pi = np.pi

'''c_profile means cumulative profile. 
It always starts at 0 at top-of-canopy and is non-decreasing.
It contains 1 more entry than the d_profile, which is layer thickness.'''


def calc_c_profile(d_profile):
    c_profile = [0]

    for d in d_profile:
        c_profile.append(c_profile[-1] + d)

    return np.array(c_profile)


class CanopyComponent:

    def __init__(self, label, d_profile, r, t,
                 r_phase=Phase.isotropic,
                 t_phase=Phase.isotropic,
                 g_orient=Orient.spherical):
        d_profile = np.array(d_profile)
        assert d_profile.all() >= 0, "Layer thicknesses should be non-negative"
        self.d_profile = d_profile
        self.c_profile = calc_c_profile(self.d_profile)

        self.label = str(label)

        # valid reflectance and transmittance
        r = np.array(r)
        t = np.array(t)
        assert len(r) == len(t), "Ensure reflectance and transmittance values are for the same set of wavebands"
        a = 1 - (r + t)

        for i in range(len(r)):
            assert 0 <= r[i] <= 1, "Reflectance values must be between 0 and 1"
            assert 0 <= t[i] <= 1, "Transmittance values must be between 0 and 1"
            assert 0 <= a[i] <= 1, "Reflectance and transmittance values must sum to between 0 and 1"

        self.r = r
        self.t = t
        self.a = a  # absorbance

        self.r_phase = r_phase
        self.t_phase = t_phase
        self.g_orient = g_orient

    def get_string(self):
        return self.label

    def __str__(self):
        return self.get_string()

    def get_d_profile(self):
        return self.d_profile
