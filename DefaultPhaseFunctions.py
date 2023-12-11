import numpy as np
import scipy.integrate as integrate

pi = np.pi


def isotropic(theta_i, theta_o, d_phi):
    assert 0 <= theta_i <= pi / 2, "Angle out of range in isotropic phase function"
    assert 0 <= theta_o <= pi / 2, "Angle out of range in isotropic phase function"
    assert 0 <= d_phi <= pi, "Angle out of range in isotropic phase function"
    return 1 / pi


# psuedo-Henyey-Greenstein phase function
def psuedo_HG(theta_i, theta_o, d_phi, g):
    assert 0 <= theta_i <= pi / 2, "Angle out of range in isotropic phase function"
    assert 0 <= theta_o <= pi / 2, "Angle out of range in isotropic phase function"
    assert 0 <= d_phi <= pi, "Angle out of range in isotropic phase function"
    assert -0.8 <= g <= 0.8, "Choose smaller g in psuedo_HG for better results"

    v_in = np.array([np.sin(theta_i), 0, np.cos(theta_i)])
    v1_out = np.array([np.sin(theta_o) * np.cos(d_phi), np.sin(theta_o) * np.sin(d_phi), np.cos(theta_o)])
    mu1 = np.dot(v_in, v1_out)
    val1 = 1 / (2 * pi) * (1 - g ** 2) / ((1 + g ** 2 - 2 * g * mu1) ** (3 / 2))

    v2_out = v1_out * np.array([1, 1, -1])  # flipped across the surface
    mu2 = np.dot(v_in, v2_out)
    val2 = 1 / (2 * pi) * (1 - g ** 2) / ((1 + g ** 2 - 2 * g * mu2) ** (3 / 2))

    val = val1 + val2
    return val


def quantify_forward_scatter(phase_fn):
    # shoot a beam at the leaf at pi/3 radians off the surface normal
    # integrate over the quantity which is forward scattered to get a rough idea of the forward scattering

    forward_scatter_fraction = integrate.nquad(
        lambda theta_i, theta_o, d_phi: phase_fn(theta_i, theta_o, d_phi) * np.sin(theta_o) * np.sin(theta_i),
        [[0, pi / 2], [0, pi / 2], [0, pi / 2]])[0]

    strong_forward_scatter = integrate.nquad(
        lambda theta_i, theta_o, d_phi: phase_fn(theta_i, theta_o, d_phi) * np.sin(theta_o) * np.sin(theta_i) * np.cos(
            theta_i - theta_o)**2 * np.cos(d_phi)**2,
        [[0, pi / 2], [0, pi / 2], [0, pi]])[0]
    return forward_scatter_fraction, strong_forward_scatter


'''
a = integrate.nquad(lambda theta_o, d_phi: psuedo_HG(1 / 2, theta_o, d_phi, 0) * np.sin(theta_o), [[0, pi / 2], [0, pi]])[0]
print(a)

g = -0.7
a, b = quantify_forward_scatter(lambda a, b, c: psuedo_HG(a, b, c, g))
print(a)
print(b)
'''

