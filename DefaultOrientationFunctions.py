import numpy as np
import scipy.integrate as integrate

pi = np.pi
epsilon = 10 ** (-8)


# mu is cosine of angle off of vertical, normal to the oriented plane
def spherical(mu):
    assert -1 <= mu <= 1, "Angle out of range in spherical orientation function"
    return 1 / 2


#  planophile is more horizontal
def planophile(mu):
    assert -1 <= mu <= 1, "Angle out of range in planophile orientation function"

    if abs(mu) > 1 - epsilon:
        return 0

    theta = np.arccos(abs(mu))
    theta = pi / 2 - theta

    return 1 / pi * (1 + np.cos(2 * theta)) / (np.sqrt(1 - mu ** 2))


# uniform in theta
def uniform(mu):
    assert -1 <= mu <= 1, "Angle out of range in uniform orientation function"

    # avoid /0 error
    if abs(mu) > 1 - epsilon:
        return 2250

    return 1 / pi / (np.sqrt(1 - mu ** 2))


# favors vertical
def erectophile(mu):
    assert -1 <= mu <= 1, "Angle out of range in erectophile orientation function"

    if abs(mu) > 1 - epsilon:
        return 4500

    theta = np.arccos(abs(mu))
    theta = pi / 2 - theta

    return 1 / pi * (1 - np.cos(2 * theta)) / (np.sqrt(1 - mu ** 2))


# favors 45 degree angle, few horizontal or vertical
def plagiophile(mu):
    assert -1 <= mu <= 1, "Angle out of range in plagiophile orientation function"

    if abs(mu) > 1 - epsilon:
        return 0

    theta = np.arccos(abs(mu))
    theta = pi / 2 - theta

    return 1 / pi * (1 - np.cos(4 * theta)) / (np.sqrt(1 - mu ** 2))


# mu is cosine of angle off of vertical, normal to the oriented plane
# x is a parameter for ellipsoidal-ness
# x < 1 means leaves more vertical, x > 1 means leaves more horizontal
# x=1 recovers spherical
def ellipsoidal(mu, x):
    assert -1 <= mu <= 1, "Angle out of range in ellipsoidal orientation function"

    #theta_l = np.arccos(abs(mu))
    #theta_l = pi / 2 - theta_l

    if x < 1:
        e1 = np.sqrt(1 - x ** 2)
        l = x + np.arcsin(e1) / e1

        # avoid \0 error
        if abs(mu) > 1 - epsilon:
            return x ** 3 / l  # avoid /0 error

    elif x > 1:
        e2 = np.sqrt(1 - x ** -2)
        l = x + np.log((1 + e2) / (1 - e2)) / (2 * e2 * x)

        # avoid \0 error
        if abs(mu) > 1 - epsilon:
            return x ** 3 / l  # avoid /0 error

    elif x == 1:
        # spherical
        return 1 / 2

    p1 = x ** 3 * abs(mu)
    p2 = (1 - mu ** 2 + x ** 2 * mu ** 2) ** 2

    return p1 / (l * p2) / (np.sqrt(1 - mu ** 2))


# returns mean, standard deviation, skewness of the leaf orientation function (just in 0-pi/2, so the mean isn't 0)
def mla_std_skew_top_half(g):
    mla = integrate.quad(lambda mu: (pi / 2 - np.arccos(mu)) * 2 * g(mu), 0, 1)[0]
    var = integrate.quad(lambda mu: ((pi / 2 - np.arccos(mu)) - mla) ** 2 * 2 * g(mu), 0, 1)[0]
    std = np.sqrt(var)
    skew = integrate.quad(lambda mu: ((pi / 2 - np.arccos(mu)) - mla) ** 3 * 2 * g(mu), 0, 1)[0] / std ** 3
    return mla, std, skew


'''
# check it integrates to 0
x = 1 / 2
a = integrate.quad(lambda mu: planophile(mu), -1, 1)[0]
print(a)

# get mean leaf angle, variance of leaf angle using only the top half of the distribution
mla, std, skew = mla_std_skew_top_half(lambda mu: planophile(mu))
print(mla * 180 / pi)
print(std * 180 / pi)
print(skew * 180 / pi)
'''
