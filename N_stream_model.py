import time
import numpy as np
import Canopy as C
import CanopyComponent as CC
import IncomingRadiation as IR
import DefaultOrientationFunctions as Orient
import DefaultPhaseFunctions as Phase
import warnings
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import xarray as xr

pi = np.pi


# all streams should have the same area when projected onto the surface of a sphere
def get_stream_division_mus(n):
    area = 4 * pi / n
    mus = [1]

    warnings.simplefilter("ignore")
    for i in range(n):
        # angles.append(arccos(np.cos(angles[-1]) - area / (2 * pi)))
        mus.append(mus[-1] - area / (2 * pi))
    warnings.simplefilter("default")

    # sometimes get numerical problems on domain of arccos, but last value should be pi
    mus[-1] = -1

    for i in mus:
        assert not np.isnan(i), "Numerical error when defining stream boundary angles"

    return mus


def arccos(x):
    return np.arccos(round(x, 8))


def spherical_to_xyz(theta_phi):
    return np.array([np.sin(theta_phi[0]) * np.cos(theta_phi[1]),
                     np.sin(theta_phi[0]) * np.sin(theta_phi[1]),
                     np.cos(theta_phi[0])
                     ])


def G_fn(g_fn, mu, precision, tol):
    """
    # old direct implementation
    theta = arccos(mu)

    mc_mu_samples = np.linspace(-1, 1, num=precision)
    mc_phi_samples = np.linspace(0, pi, num=precision)
    results = np.zeros(precision ** 2)

    theta_hat = spherical_to_xyz([theta, 0])  # incoming beam
    for i, mu_p in enumerate(mc_mu_samples):
        theta_p = arccos(mu_p)
        for j, phi_p in enumerate(mc_phi_samples):
            value = g_fn(mu_p)
            n_hat = spherical_to_xyz([theta_p, phi_p])  # surface normal
            value *= np.abs(np.dot(theta_hat, n_hat))
            results[i * precision + j] = value

    G = np.average(results) * 2
    """

    # scipy implementation
    v_hat = spherical_to_xyz([arccos(mu), 0])

    def inside_integral_G_fn(mu_p):
        def inside_integral_2_G_fn(phi_p):
            n_hat = spherical_to_xyz([arccos(mu_p), phi_p])
            dot = np.dot(v_hat, n_hat)
            if dot > 0:
                return 0
            return -dot

        return g_fn(mu_p) * integrate.quad(lambda phi_p: inside_integral_2_G_fn(phi_p), 0, pi, epsrel=tol)[0]

    G = 1 / pi * integrate.quad(lambda mu_var: inside_integral_G_fn(mu_var), -1, 1, epsrel=tol)[0]

    return G


def aa_phase_fn(rt, mu_i, mu_j, g_fn, phase_fn, precision):
    numerator = aa_phase_fn_numerator(rt, mu_i, mu_j, g_fn, phase_fn, precision)
    denominator = aa_phase_fn_denominator(rt, mu_i, g_fn, phase_fn, precision)
    return numerator / denominator


def aa_phase_fn_inside_integral_scipy(rt, mu_i, mu_j_min, mu_j_max, g_fn, phase_fn, precision, tol):
    denominator = aa_phase_fn_denominator(rt, mu_i, g_fn, phase_fn, precision, tol)

    numerator = integrate.quad(
        lambda mu_j_var: G_fn(g_fn, mu_j_var, precision, tol) *
                         aa_phase_fn_numerator(rt, mu_i, mu_j_var, g_fn, phase_fn, precision, tol),
        mu_j_min, mu_j_max, epsrel=tol)[0]

    return numerator / denominator


def aa_phase_fn_numerator(rt, mu_i, mu_j, g_fn, phase_fn, precision, tol):
    if rt == 'r':
        rt = -1
    elif rt == 't':
        rt = 1

    # monte-carlo implementation
    theta_in = arccos(mu_i)
    theta_out = arccos(mu_j)
    theta_in_hat = spherical_to_xyz([theta_in, 0])  # incoming beam

    mc_mu_p_samples = np.linspace(-1, 1, num=precision * 2 + 1)[1:-1:2]
    mc_phi_p_samples = np.linspace(0, pi, num=precision * 2 + 1)[1:-1:2]
    mc_phi_out_samples = np.linspace(0, 2 * pi, num=2 * precision, endpoint=False)

    results = np.zeros((precision, precision, 2 * precision))

    # set up triple integral
    for i, mu_p in enumerate(mc_mu_p_samples):

        g = g_fn(np.cos(mu_p))

        for j, phi_p in enumerate(mc_phi_p_samples):

            n_hat = spherical_to_xyz([arccos(mu_p), phi_p])  # surface normal

            # check incoming beam is approaching surface from correct direction
            if np.dot(theta_in_hat, n_hat) >= 0:
                results[i, j, :] = 0
                continue

            G = np.abs(np.dot(theta_in_hat, n_hat))

            # compute phase_theta_in, theta_in for the phase function
            phase_theta_in = pi / 2 - arccos(G)

            for k, phi_out in enumerate(mc_phi_out_samples):

                theta_out_hat = spherical_to_xyz([theta_out, phi_out])

                # check outgoing beam is leaving surface from correct direction
                if np.dot(theta_out_hat, n_hat) * rt >= 0:
                    results[i, j, k] = 0
                    continue

                # compute phase_theta_out, theta_out for the phase function
                phase_theta_out = pi / 2 - arccos(np.abs(np.dot(theta_out_hat, n_hat)))

                # compute phase_d_phi, delta_phi for the phase function
                # project incoming and exiting beams onto surface, normalize, find angle between
                projected_in = theta_in_hat - np.dot(theta_in_hat, n_hat) * n_hat
                projected_out = theta_out_hat - np.dot(theta_out_hat, n_hat) * n_hat

                norm_in = np.linalg.norm(projected_in)
                norm_out = np.linalg.norm(projected_out)
                if norm_in == 0 or norm_out == 0:
                    # 0% chance of happening
                    results[i, j, k] = 0
                    continue

                projected_in /= norm_in
                projected_out /= norm_out

                phase_d_phi = arccos(np.dot(projected_in, projected_out))

                value = phase_fn(phase_theta_in, phase_theta_out, phase_d_phi)

                value = value * g * G

                results[i, j, k] = value

    p = np.average(results) * 4 * pi ** 2

    # scipy implementation 1 (2 scipy integrals, one monte-carlo)
    '''
    theta_in = arccos(mu_i)
    theta_out = arccos(mu_j)
    theta_in_hat = spherical_to_xyz([theta_in, 0])  # incoming beam

    mc_phi_out_samples = np.linspace(0, 2 * pi, num= precision, endpoint=False)

    # set up triple integral
    def inside_integral_1(mu_p):

        g = g_fn(np.cos(mu_p))

        def inside_integral_2(phi_p):

            n_hat = spherical_to_xyz([arccos(mu_p), phi_p])  # surface normal

            # check incoming beam is approaching surface from correct direction
            if np.dot(theta_in_hat, n_hat) >= 0:
                return 0

            G = np.abs(np.dot(theta_in_hat, n_hat))

            # compute phase_theta_in, theta_in for the phase function
            phase_theta_in = pi / 2 - arccos(G)

            def inside_integral_3():

                results = np.zeros(precision)

                for i, phi_out in enumerate(mc_phi_out_samples):

                    theta_out_hat = spherical_to_xyz([theta_out, phi_out])

                    # check outgoing beam is leaving surface from correct direction
                    if np.dot(theta_out_hat, n_hat) * rt >= 0:
                        results[i] = 0
                        continue

                    # compute phase_theta_out, theta_out for the phase function
                    phase_theta_out = pi / 2 - arccos(np.abs(np.dot(theta_out_hat, n_hat)))

                    # compute phase_d_phi, delta_phi for the phase function
                    # project incoming and exiting beams onto surface, normalize, find angle between
                    projected_in = theta_in_hat - np.dot(theta_in_hat, n_hat) * n_hat
                    projected_out = theta_out_hat - np.dot(theta_out_hat, n_hat) * n_hat

                    norm_in = np.linalg.norm(projected_in)
                    norm_out = np.linalg.norm(projected_out)
                    if norm_in == 0 or norm_out == 0:
                        # 0% chance of happening
                        results[i] = 0
                        continue

                    projected_in /= norm_in
                    projected_out /= norm_out

                    phase_d_phi = arccos(np.dot(projected_in, projected_out))

                    results[i] = phase_fn(phase_theta_in, phase_theta_out, phase_d_phi)

                return np.average(results) * 2 * pi

            return G * inside_integral_3()

        return g * integrate.quad(lambda phi_p_var: inside_integral_2(phi_p_var), 0, pi, epsrel=tol)[0]

    p = integrate.quad(lambda mu_p_var: inside_integral_1(mu_p_var), -1, 1, epsrel=tol)[0]
    
    return p
    '''
    '''
    # scipy implementation 2 (1 scipy integral, 2 monte-carlo). Performs the best
    theta_in = arccos(mu_i)
    theta_out = arccos(mu_j)
    theta_in_hat = spherical_to_xyz([theta_in, 0])  # incoming beam

    mc_phi_p_samples = np.linspace(0, pi, num=precision * 2 + 1)[1:-1:2]
    mc_phi_out_samples = np.linspace(0, 2 * pi, num=precision, endpoint=False)

    # set up triple integral
    # scipy on the outside integral, montecarlo on the inside 2
    def inside_integral_1(mu_p):

        g = g_fn(np.cos(mu_p))

        results = np.zeros((precision, 2 * precision))

        for j, phi_p in enumerate(mc_phi_p_samples):

            n_hat = spherical_to_xyz([arccos(mu_p), phi_p])  # surface normal

            # check incoming beam is approaching surface from correct direction
            if np.dot(theta_in_hat, n_hat) >= 0:
                results[j, :] = 0
                continue

            G = np.abs(np.dot(theta_in_hat, n_hat))

            # compute phase_theta_in, theta_in for the phase function
            phase_theta_in = pi / 2 - arccos(G)

            for k, phi_out in enumerate(mc_phi_out_samples):

                theta_out_hat = spherical_to_xyz([theta_out, phi_out])

                # check outgoing beam is leaving surface from correct direction
                if np.dot(theta_out_hat, n_hat) * rt >= 0:
                    results[j, k] = 0
                    continue

                # compute phase_theta_out, theta_out for the phase function
                phase_theta_out = pi / 2 - arccos(np.abs(np.dot(theta_out_hat, n_hat)))

                # compute phase_d_phi, delta_phi for the phase function
                # project incoming and exiting beams onto surface, normalize, find angle between
                projected_in = theta_in_hat - np.dot(theta_in_hat, n_hat) * n_hat
                projected_out = theta_out_hat - np.dot(theta_out_hat, n_hat) * n_hat

                norm_in = np.linalg.norm(projected_in)
                norm_out = np.linalg.norm(projected_out)
                if norm_in == 0 or norm_out == 0:
                    # 0% chance of happening
                    results[j, k] = 0
                    continue

                projected_in /= norm_in
                projected_out /= norm_out

                phase_d_phi = arccos(np.dot(projected_in, projected_out))

                value = phase_fn(phase_theta_in, phase_theta_out, phase_d_phi)

                value = value * g * G

                results[j, k] = value

        return np.average(results) * 2 * pi ** 2

    p = integrate.quad(lambda mu_p_var: inside_integral_1(mu_p_var), -1, 1, epsrel=tol)[0]
    '''
    return p


def aa_phase_fn_denominator(rt, mu_i, g_fn, phase_fn, precision, tol):
    """
    # old direct implementation
    mc_mu_j_samples = np.linspace(-1, 1, num=precision)
    results = np.zeros(precision)
    # set up integral
    for i, mu_j in enumerate(mc_mu_j_samples):
        value = aa_phase_fn_numerator(rt, mu_i, mu_j, g_fn, phase_fn, precision)
        results[i] = value
    Z = np.average(results) * 2
    """

    Z = integrate.quad(lambda mu_j_var: aa_phase_fn_numerator(rt, mu_i, mu_j_var, g_fn, phase_fn, precision, tol),
                       -1, 1, epsrel=tol)[0]

    return Z


def d_something_dz(zs, d_profile, c_z, d_z):
    n_layers = len(d_z)
    a = np.zeros(len(zs))

    for i, z in enumerate(zs):
        for j in range(n_layers):
            if c_z[j + 1] >= z:
                break
        a[i] = d_profile[j] / d_z[j]

    return a


def c_something(zs, c_profile, d_profile, c_z, d_z):
    n_layers = len(d_z)
    a = np.zeros(len(zs))

    for i, z in enumerate(zs):
        for j in range(n_layers):
            if c_z[j + 1] >= z:
                diff = c_z[j + 1] - z
                break
        a[i] = c_profile[j + 1] - d_profile[j] / d_z[j] * diff

    # print('c: ' + str(a))
    return a


def get_wb_centers_from_edges(wb_edges):
    wb_centers = np.zeros(len(wb_edges) - 1)

    for i in range(len(wb_centers)):
        wb_centers[i] = (wb_edges[i] + wb_edges[i + 1]) / 2

    return wb_centers


def get_wb_widths_from_edges(wb_edges):
    wb_widths = np.zeros(len(wb_edges) - 1)

    for i in range(len(wb_widths)):
        wb_widths[i] = wb_edges[i + 1] - wb_edges[i + 1]

    return wb_widths


class NStreamModel:

    def __init__(self, n_streams, n_layers, n_wb):
        assert n_streams % 2 == 0, "Must have even number of streams"
        self.n_streams = n_streams
        self.mus = get_stream_division_mus(self.n_streams)
        self.n_layers = n_layers
        self.n_wb = n_wb
        self.wb_edges = None
        self.wb_centers = None
        self.wb_widths = None
        self.canopy = None
        self.incoming_radiation = None
        self.geometry_factors = None
        self.has_ran = False
        self.bvp_solutions = None
        self.direct_beam = None

        self.extras = {'down_flux_from_dif': None,
                       'up_flux_from_dif': None,
                       'net_flux_from_dif': None,
                       'down_flux_from_dir': None,
                       'up_flux_from_dir': None,
                       'net_flux_from_dir': None,
                       'net_flux': None,
                       'actinic_flux_from_dif': None,
                       'actinic_flux_from_dir': None,
                       'actinic_flux': None}

    def set_wb_edges(self, wb):
        wb_edges = np.array(wb)

        assert len(wb_edges) == self.n_wb + 1, "Set waveband edges with set_wb"
        assert wb_edges[0] > 0, "Wavelengths should be positive"
        for i in range(self.n_wb):
            assert wb_edges[i] < wb_edges[i + 1], "Waveband edges should be strictly increasing"

        self.wb_edges = wb_edges
        self.wb_centers = get_wb_centers_from_edges(self.wb_edges)
        self.wb_widths = get_wb_widths_from_edges(self.wb_edges)

        assert len(self.wb_centers) == self.n_wb
        assert len(self.wb_widths) == self.n_wb

    def add_canopy(self, canopy):
        assert type(canopy) is C.Canopy, "Object must be of type Canopy"
        assert canopy.n_wb == self.n_wb, "Ensure number of wavebands are consistent"
        assert len(canopy.components) != 0, "You must have at least one CanopyComponent"
        assert canopy.soil_r is not None, "Must add soil reflectance to canopy with set_soil()"
        assert len(canopy.dz) == self.n_layers, "Must have correct number of layers in canopy"

        self.canopy = canopy

    def add_incoming_radiation(self, incoming_rad):
        assert type(incoming_rad) is IR.IncomingRadiation
        assert incoming_rad.n_wb == self.n_wb
        assert incoming_rad.direct is not None
        assert incoming_rad.diffuse is not None

        self.incoming_radiation = incoming_rad

    def __str__(self):
        string = str(self.n_streams) + '-stream model with ' + str(self.n_wb) + ' wavebands.\n'
        if self.canopy is not None:
            string += self.canopy.get_string()
        return string

    def plot_ins(self):
        if self.canopy is not None:
            for component in self.canopy.components:
                plt.plot(component.d_profile / self.canopy.dz, self.canopy.z_centers)
                plt.title(str(component.label) + " vertical distribution")
                plt.xlabel(r"Area Index $\left(\frac{m^2}{m^2}/m\right)$")
                plt.ylabel(r"Canopy Depth $(m)$")
                plt.gca().invert_yaxis()
                plt.show()
                plt.plot(component.c_profile, self.canopy.z)
                plt.title(str(component.label) + " cumulative vertical distribution")
                plt.ylabel(r"Canopy Depth $(m)$")
                plt.xlabel(r"Area Index $\left(\frac{m^2}{m^2}\right)$")
                plt.gca().invert_yaxis()
                plt.show()
                if self.wb_centers is not None:
                    plt.plot(self.wb_centers, component.r, label='r')
                    plt.plot(self.wb_centers, component.t, label='t')
                    plt.plot(self.wb_centers, component.r + component.t, label='ssa')
                    plt.title(str(component.label) + " optical properties")
                    plt.xlabel("nm")
                    plt.ylim([-0.1, 1.1])
                    plt.legend()
                    plt.show()
            if self.wb_centers is not None:
                plt.plot(self.wb_centers, self.canopy.soil_r)
                plt.title("Soil Reflectivity")
                plt.xlabel("nm")
                plt.ylim([-0.1, 1.1])
                plt.show()
        if self.incoming_radiation is not None and self.wb_centers is not None:
            plt.plot(self.wb_centers, self.incoming_radiation.direct, label='direct')
            plt.plot(self.wb_centers, self.incoming_radiation.diffuse, label='diffuse')
            plt.title(r"Incoming Radiation $\frac{W}{m^2}$")
            plt.legend()
            plt.ylim(bottom=0)
            plt.show()

    def compute_geometry_mc(self, precision, tol):

        n = self.n_streams
        num_mc = int(precision / n) + 1

        self.geometry_factors = {}

        for component in self.canopy.components:
            d = {}

            # 1: compute all G_i's
            G_int = np.zeros(n)

            for i in range(n):
                mc_samples = np.random.uniform(self.mus[i + 1], self.mus[i], num_mc)
                results = np.zeros(num_mc)
                for j, mu in enumerate(mc_samples):
                    results[j] = G_fn(component.g_orient, mu, precision)
                G_int[i] = np.average(results) * (self.mus[i] - self.mus[i + 1]) * 2
            d["G_i"] = G_int

            # 2: compute reflection scattering matrix
            # entry in row i and column j [i,j] means light scattered to stream i from stream j
            dif_r_matrix = np.zeros((n, n))

            for i in range(n):
                for j in range(n):
                    mc_mu_i = np.random.uniform(self.mus[i + 1], self.mus[i], num_mc)
                    mc_mu_j = np.random.uniform(self.mus[j + 1], self.mus[j], num_mc)
                    results = np.zeros(num_mc ** 2)
                    for k, mu_i in enumerate(mc_mu_i):
                        denominator = aa_phase_fn_denominator('r', mu_i, component.g_orient, component.r_phase,
                                                              precision)
                        for l, mu_j in enumerate(mc_mu_j):
                            value = aa_phase_fn_numerator('r', mu_i, mu_j, component.g_orient,
                                                          component.r_phase, precision) / denominator

                            value *= G_fn(component.g_orient, mu_j, precision)
                            results[k * num_mc + l] = value
                    dif_r_matrix[i, j] = np.average(results) * (self.mus[i] - self.mus[i + 1]) * (
                            self.mus[j] - self.mus[j + 1]) * 2
            d["dif_r"] = dif_r_matrix

            # 3: compute transmission scattering matrix
            # entry in row i and column j [i,j] means light scattered to stream i from stream j
            dif_t_matrix = np.zeros((n, n))

            for i in range(n):
                for j in range(n):
                    mc_mu_i = np.random.uniform(self.mus[i + 1], self.mus[i], num_mc)
                    mc_mu_j = np.random.uniform(self.mus[j + 1], self.mus[j], num_mc)
                    results = np.zeros(num_mc ** 2)
                    for k, mu_i in enumerate(mc_mu_i):
                        denominator = aa_phase_fn_denominator('t', mu_i, component.g_orient, component.r_phase,
                                                              precision)
                        for l, mu_j in enumerate(mc_mu_j):
                            value = aa_phase_fn_numerator('t', mu_i, mu_j, component.g_orient,
                                                          component.r_phase, precision) / denominator

                            value *= G_fn(component.g_orient, mu_j, precision)
                            results[k * num_mc + l] = value
                    dif_t_matrix[i, j] = np.average(results) * (self.mus[i] - self.mus[i + 1]) * (
                            self.mus[j] - self.mus[j + 1]) * 2
            d["dif_t"] = dif_t_matrix

            # 4: compute G_0, which appears in the exponent of the direct beam scattering terms
            mu_0 = np.cos(self.incoming_radiation.sza)
            d["G_0"] = np.array([G_fn(component.g_orient, mu_0, precision)])

            # 5: compute direct beam reflection scattering integral
            dir_beam_r = np.zeros(n)

            denominator = aa_phase_fn_denominator('r', mu_0, component.g_orient, component.r_phase, precision)

            for i in range(n):
                mc_mu = np.random.uniform(self.mus[i + 1], self.mus[i], num_mc)
                results = np.zeros(num_mc)
                for j, mu in enumerate(mc_mu):
                    results[j] = aa_phase_fn_numerator('r', mu_0, mu, component.g_orient, component.r_phase,
                                                       precision) / denominator

                dir_beam_r[i] = np.average(results) * (self.mus[i] - self.mus[i + 1]) * d['G_0'][0] / pi
            d["dir_r"] = dir_beam_r

            # 6: compute direct beam transmission scattering integral
            dir_beam_t = np.zeros(n)

            denominator = aa_phase_fn_denominator('t', mu_0, component.g_orient, component.r_phase, precision)

            for i in range(n):
                mc_mu = np.random.uniform(self.mus[i + 1], self.mus[i], num_mc)
                results = np.zeros(num_mc)
                for j, mu in enumerate(mc_mu):
                    results[j] = aa_phase_fn_numerator('t', mu_0, mu, component.g_orient, component.r_phase,
                                                       precision) / denominator

                dir_beam_t[i] = np.average(results) * (self.mus[i] - self.mus[i + 1]) * d['G_0'][0] / pi
            d["dir_t"] = dir_beam_t

            self.geometry_factors[component.label] = d

        # compute the amount of direct beam at the soil
        exponent = 0
        for comp in self.canopy.components:
            exponent -= self.geometry_factors[comp.label]['G_0'][0] * comp.c_profile[-1]
        exponent /= mu_0
        exponential = np.e ** exponent
        self.geometry_factors['soil'] = {'dir_at_soil': np.array([exponential])}

    def compute_geometry_scipy(self, precision, tol):
        """
        same as compute_geometry_mc but attempts to use scipy to integrate.
        This should work well as some functions are smooth, namely the aa_phase_function and G_fn, though some
        integrals computed within those functions aren't smooth, the outer ones are.
        Hopefully this gets rid of some of the worst O(n^5) behavior and makes it more like O(n^3)
        with respect to precision tolerance
        """

        start_time = time.time()

        n = self.n_streams
        num_mc = int(precision / n) + 1

        self.geometry_factors = {}

        for component in self.canopy.components:
            d = {}

            # 1: compute all G_i's
            G_int = np.zeros(n)

            for i in range(n):
                integral = integrate.quad(lambda mu_var: G_fn(component.g_orient, mu_var, precision, tol),
                                          self.mus[i + 1], self.mus[i], epsrel=tol)[0]
                G_int[i] = integral * 2

            p = np.sum(G_int)
            if abs(p - 1) > 0.01:
                print('Projected leaf area G matrix not computed well, sum is ' + str(p))
            G_int = G_int / p

            d["G_i"] = G_int

            # print(time.time() - start_time)
            start_time = time.time()

            # 2: compute reflection scattering matrix
            # entry in row i and column j [i,j] means light scattered to stream i from stream j
            dif_r_matrix = np.zeros((n, n))

            for i in range(n):
                for j in range(n):
                    integral = integrate.quad(lambda mu_i_var: aa_phase_fn_inside_integral_scipy(
                        'r', mu_i_var, self.mus[j + 1], self.mus[j], component.g_orient, component.r_phase, precision,
                        tol), self.mus[i + 1], self.mus[i], epsrel=tol)[0]

                    dif_r_matrix[i, j] = integral * 2

                    # print(time.time() - start_time)
                    start_time = time.time()

            p = np.sum(dif_r_matrix)
            if abs(p - 1) > 0.01:
                print('Diffuse reflection scattering matrix not computed well, sum is ' + str(p))
            dif_r_matrix = dif_r_matrix / p

            d["dif_r"] = dif_r_matrix

            # print(time.time() - start_time)
            start_time = time.time()

            # 3: compute transmission scattering matrix
            # entry in row i and column j [i,j] means light scattered to stream i from stream j
            dif_t_matrix = np.zeros((n, n))

            for i in range(n):
                for j in range(n):
                    integral = integrate.quad(lambda mu_i_var: aa_phase_fn_inside_integral_scipy(
                        't', mu_i_var, self.mus[j + 1], self.mus[j], component.g_orient, component.t_phase, precision,
                        tol), self.mus[i + 1], self.mus[i], epsrel=tol)[0]

                    dif_t_matrix[i, j] = integral * 2
            p = np.sum(dif_t_matrix)
            if abs(p - 1) > 0.01:
                print('Diffuse transmission scattering matrix not computed well, sum is ' + str(p))
            dif_t_matrix = dif_t_matrix / p

            d["dif_t"] = dif_t_matrix

            # print(time.time() - start_time)
            start_time = time.time()

            # 4: compute G_0, which appears in the exponent of the direct beam scattering terms
            mu_0 = np.cos(pi - self.incoming_radiation.sza)
            d["G_0"] = np.array([G_fn(component.g_orient, mu_0, precision, tol)])

            # print(time.time() - start_time)
            start_time = time.time()

            # 5: compute direct beam reflection scattering integral
            dir_beam_r = np.zeros(n)

            denominator = aa_phase_fn_denominator('r', mu_0, component.g_orient, component.r_phase, precision, tol)

            for i in range(n):
                integral = integrate.quad(
                    lambda mu_var: aa_phase_fn_numerator('r', mu_0, mu_var, component.g_orient, component.r_phase,
                                                         precision, tol), self.mus[i + 1], self.mus[i], epsrel=tol)[0]

                dir_beam_r[i] = integral / denominator * d['G_0'][0] / pi
            d["dir_r"] = dir_beam_r

            # print(time.time() - start_time)
            start_time = time.time()

            # 6: compute direct beam transmission scattering integral
            dir_beam_t = np.zeros(n)

            denominator = aa_phase_fn_denominator('t', mu_0, component.g_orient, component.r_phase, precision, tol)

            for i in range(n):
                integral = integrate.quad(
                    lambda mu_var: aa_phase_fn_numerator('t', mu_0, mu_var, component.g_orient, component.r_phase,
                                                         precision, tol), self.mus[i + 1], self.mus[i], epsrel=tol)[0]

                dir_beam_t[i] = integral / denominator * d['G_0'][0] / pi
            d["dir_t"] = dir_beam_t

            # print(time.time() - start_time)
            start_time = time.time()

            self.geometry_factors[component.label] = d

        # compute the amount of direct beam at the soil
        exponent = 0
        for comp in self.canopy.components:
            exponent -= self.geometry_factors[comp.label]['G_0'][0] * comp.c_profile[-1]
        exponent /= mu_0
        exponential = np.e ** exponent
        self.geometry_factors['soil'] = {'dir_at_soil': np.array([exponential])}

    def dIdz_RHS(self, z, I, wb, direct):
        ''' RHS of differential equation
        z: geometric height
        I: [the n_streams' intensities at z]
        wb: current wb (for r and t)
        direct: if not direct we can ignore some terms
        '''
        # print('z: ' + str(z))
        # print('I: ' + str(I))
        results = np.zeros((self.n_streams, len(z)))

        if direct:
            mu_0 = np.cos(self.incoming_radiation.sza)
            exponent = 0
            for comp in self.canopy.components:
                exponent -= self.geometry_factors[comp.label]['G_0'][0] * c_something(z, comp.c_profile,
                                                                                      comp.d_profile, self.canopy.z,
                                                                                      self.canopy.dz)
            exponent /= mu_0
            exponential = np.e ** exponent

        for i in range(self.n_streams):
            result = 0
            for component in self.canopy.components:

                sink = I[i] * self.geometry_factors[component.label]['G_i'][i]

                dif_r = 0
                for j in range(self.n_streams):
                    dif_r += I[j] * self.geometry_factors[component.label]['dif_r'][i, j]

                dif_t = 0
                for j in range(self.n_streams):
                    dif_t += I[j] * self.geometry_factors[component.label]['dif_t'][i, j]

                dir_r = 0
                dir_t = 0
                if direct:
                    dir_r = 2 * pi * exponential * self.geometry_factors[component.label]['dir_r'][i]
                    dir_t = 2 * pi * exponential * self.geometry_factors[component.label]['dir_t'][i]

                component_result = -sink + component.r[wb] * (dif_r + dir_r) + component.t[wb] * (dif_t + dir_t)
                component_result *= d_something_dz(z, component.d_profile, self.canopy.z, self.canopy.dz)
                result += component_result
            result /= -(self.mus[i] ** 2 - self.mus[i + 1] ** 2)
            results[i] = result

        # print('dI/dz: ' + str(results))
        return results

    def calc_bcs(self, toc, boc, wb, direct):
        # print('toc: ' + str(toc))
        # print('boc: ' + str(boc))

        residuals = np.zeros(self.n_streams)

        half = int(self.n_streams / 2)

        # first half of streams (upward) have a bc at soil
        # we assume isotropic reflection off of soil surface from streams and direct beam
        downward_flux = 0
        for i in range(half, self.n_streams):
            downward_flux += boc[i] * (self.mus[i + 1] ** 2 - self.mus[i] ** 2)
        downward_flux *= pi

        if direct:
            downward_flux += self.direct_beam[-1] * np.cos(self.incoming_radiation.sza)

        reflected_flux = downward_flux * self.canopy.soil_r[wb]
        reflected_isotropic_intensity = reflected_flux / pi

        for i in range(half):
            residuals[i] = boc[i] - reflected_isotropic_intensity

        # second half of streams (downward) have a bc at toc due to diffuse
        if direct:
            for i in range(half, self.n_streams):
                residuals[i] = toc[i]

        if not direct:
            for i in range(half, self.n_streams):
                residuals[i] = toc[i] - 1

        # print('residuals: ' + str(residuals))
        return residuals

    def run(self, precision=6, tol=1):
        '''
        :param precision: how many monte-carlo integration points to do in the innermost integration of the
        phase-function
        :param tol: relative error each scipy integral calculator uses. 1 is default and seems high, but yields high
        precision in practice, to a few decimal places
        :return: nothing. Just updates some model variables, namely the solution ones
        '''
        assert self.canopy is not None
        assert self.incoming_radiation is not None

        start_time = time.time()

        '''compute the geometry factors'''
        # self.compute_geometry_mc(precision, tol)
        self.compute_geometry_scipy(precision, tol)

        print(self.geometry_factors)

        """
        for component, component_dict in self.geometry_factors.items():
            for key, val in component_dict.items():
                a = np.ndarray.flatten(val)
                for i in a:
                    assert 0 < i <= 2, "Ensure precision is high enough to get accurate Monte-Carlo integrals"
        """

        elapsed_time = round(time.time() - start_time, 2)
        print('Computed geometry factors in ' + str(elapsed_time) + ' seconds')

        # compute direct beam strength
        # array dimension are [layers]
        mu_0 = np.cos(self.incoming_radiation.sza)
        exponent = 0
        for comp in self.canopy.components:
            exponent -= self.geometry_factors[comp.label]['G_0'][0] * c_something(self.canopy.z, comp.c_profile,
                                                                                  comp.d_profile, self.canopy.z,
                                                                                  self.canopy.dz)
        exponent /= mu_0
        self.direct_beam = np.e ** exponent

        # dimension are [waveband, direct or diffuse, streams, layers]
        bvp_solutions = np.zeros((self.n_wb, 2, self.n_streams, self.n_layers + 1))

        # run each waveband and numerically solve the 2n-coupled BVPs
        start_wb_time = time.time()
        for wb in range(self.n_wb):

            # direct beam first
            # set up the x mesh
            x = self.canopy.z

            # initial guess
            if wb == 0:
                y0 = np.ones((self.n_streams, x.size))
            else:
                y0 = bvp_solutions[wb - 1, 0]

            # fn = lambda z, I: self.dIdz_RHS(z, I, wb, direct=True)
            # bcs = lambda toc_I, boc_I: self.calc_bcs(toc_I, boc_I, wb, direct=True)

            result = integrate.solve_bvp(lambda z, I: self.dIdz_RHS(z, I, wb, direct=True),
                                         lambda toc_I, boc_I: self.calc_bcs(toc_I, boc_I, wb, direct=True), x, y0,
                                         tol=tol)
            # print(result)

            y = result.sol(self.canopy.z)  # solution interpolated to z values
            bvp_solutions[wb, 0] = y

            # print(y)

            # direct beam first
            # set up the x mesh
            x = self.canopy.z

            # initial guess (use the previous waveband's solution)
            if wb == 0:
                y0 = np.ones((self.n_streams, x.size))
            else:
                y0 = bvp_solutions[wb - 1, 1]

            # fn = lambda z, I: self.dIdz_RHS(z, I, wb, direct=True)
            # bcs = lambda toc_I, boc_I: self.calc_bcs(toc_I, boc_I, wb, direct=True)

            result = integrate.solve_bvp(lambda z, I: self.dIdz_RHS(z, I, wb, direct=False),
                                         lambda toc_I, boc_I: self.calc_bcs(toc_I, boc_I, wb, direct=False), x, y0,
                                         tol=10 ** (-6))
            # print(result)

            y = result.sol(self.canopy.z)  # solution splined over z vals
            bvp_solutions[wb, 1] = y

            # print(y)
            if wb == 0:
                print(time.time() - start_wb_time)

        self.bvp_solutions = bvp_solutions
        # print(bvp_solutions)

        elapsed_time = round(time.time() - start_time - elapsed_time, 2)
        print('Solved ' + str(self.n_wb) + ' BVPs in ' + str(elapsed_time) + ' seconds')

        self.has_ran = True

        elapsed_time = round(time.time() - start_time, 2)
        print('Done! ' + str(elapsed_time) + ' seconds')

    def compute_extras(self):
        assert self.has_ran, "Must use .run() before this"

        # Compute irradiance, actinic flux for each layer boundary and waveband from self.bvp_solutions and direct beam
        # Dimension are [layer, wb] for each quantity
        half = int(self.n_streams / 2)

        down_flux_from_dif = np.zeros((self.n_layers + 1, self.n_wb,))
        for j in range(self.n_layers + 1):
            for i in range(self.n_wb):
                flux = 0
                # second half of streams goes down
                for k in range(half, self.n_streams):
                    flux += abs(self.incoming_radiation.diffuse[i] / pi * self.bvp_solutions[i, 1, k, j] * (
                            self.mus[k + 1] ** 2 - self.mus[k] ** 2) / 2) * 2 * pi

                down_flux_from_dif[j, i] = flux
        self.extras['down_flux_from_dif'] = abs(down_flux_from_dif)

        up_flux_from_dif = np.zeros((self.n_layers + 1, self.n_wb,))
        for j in range(self.n_layers + 1):
            for i in range(self.n_wb):
                flux = 0

                # first half of streams goes up
                for k in range(half):
                    flux += abs(self.incoming_radiation.diffuse[i] / pi * self.bvp_solutions[i, 1, k, j] * (
                            self.mus[k + 1] ** 2 - self.mus[k] ** 2) / 2) * 2 * pi

                up_flux_from_dif[j, i] = flux
        self.extras['up_flux_from_dif'] = abs(up_flux_from_dif)

        self.extras['net_flux_from_dif'] = abs(down_flux_from_dif - up_flux_from_dif)

        down_flux_from_dir = np.zeros((self.n_layers + 1, self.n_wb,))
        for j in range(self.n_layers + 1):
            for i in range(self.n_wb):
                flux = 0

                # second half of streams goes down
                for k in range(half, self.n_streams):
                    flux += abs(self.incoming_radiation.direct[i] * self.bvp_solutions[i, 0, k, j] * (
                            self.mus[k + 1] ** 2 - self.mus[k] ** 2) / 2)

                flux += self.incoming_radiation.direct[i] * self.direct_beam[j]
                down_flux_from_dir[j, i] = flux
        self.extras['down_flux_from_dir'] = abs(down_flux_from_dir)

        up_flux_from_dir = np.zeros((self.n_layers + 1, self.n_wb,))
        for j in range(self.n_layers + 1):
            for i in range(self.n_wb):
                flux = 0

                # first half of streams goes up
                for k in range(half):
                    flux += abs(self.incoming_radiation.direct[i] * self.bvp_solutions[i, 0, k, j] * (
                            self.mus[k + 1] ** 2 - self.mus[k] ** 2) / 2)

                up_flux_from_dir[j, i] = flux
        self.extras['up_flux_from_dir'] = abs(up_flux_from_dir)

        self.extras['net_flux_from_dir'] = abs(down_flux_from_dir - up_flux_from_dir)

        self.extras['net_flux'] = abs(self.extras['net_flux_from_dif'] + self.extras['net_flux_from_dir'])

        '''
        self.net_flux = np.zeros((self.n_layers + 1, self.n_wb,))
        half = int(self.n_streams / 2)

        for j in range(self.n_layers + 1):
            for i in range(self.n_wb):
                net_flux = 0

                # first half of streams goes up
                for k in range(half):
                    net_flux -= abs(self.incoming_radiation.direct[i] * self.bvp_solutions[i, 0, k, j] * (
                            self.mus[k + 1] ** 2 - self.mus[k] ** 2) / 2)
                    net_flux -= abs(self.incoming_radiation.diffuse[i] / pi * self.bvp_solutions[i, 1, k, j] * (
                            self.mus[k + 1] ** 2 - self.mus[k] ** 2) / 2) * 2 * pi

                # second half of streams goes down
                for k in range(half, self.n_streams):
                    net_flux += abs(self.incoming_radiation.direct[i] * self.bvp_solutions[i, 0, k, j] * (
                            self.mus[k + 1] ** 2 - self.mus[k] ** 2) / 2)
                    net_flux += abs(self.incoming_radiation.diffuse[i] / pi * self.bvp_solutions[i, 1, k, j] * (
                            self.mus[k + 1] ** 2 - self.mus[k] ** 2) / 2) * 2 * pi

                net_flux += self.incoming_radiation.direct[i] * self.direct_beam[j]
                self.net_flux[j, i] = net_flux
                
        # check that net_flux isn't negative, which it can be due to numerical issues
        max_error = 0
        for z in self.net_flux:
            for j in range(len(z)):
                if z[j] < 0:
                    if z[j] < max_error:
                        max_error = z[j]
                    z[j] = 0
        if max_error < 0:
            print('ERROR, net_flux value of ' + str(max_error) + ' computed, set to 0')
        '''

        '''
        # Irradiance (W/m^2) total
        # Dimension are [layer, wb]
        self.flux = np.zeros((self.n_layers + 1, self.n_wb,))

        for j in range(self.n_layers + 1):
            for i in range(self.n_wb):
                flux = 0

                for k in range(0, self.n_streams):
                    flux += abs(self.incoming_radiation.direct[i] * self.bvp_solutions[i, 0, k, j] * (
                            self.mus[k + 1] ** 2 - self.mus[k] ** 2) / 2)
                    flux += abs(self.incoming_radiation.diffuse[i] / pi * self.bvp_solutions[i, 1, k, j] * (
                            self.mus[k + 1] ** 2 - self.mus[k] ** 2) / 2) * 2 * pi

                flux += self.incoming_radiation.direct[i] * self.direct_beam[j]
                self.flux[j, i] = flux
        '''

        # Actinic flux

        actinic_flux_from_dif = np.zeros((self.n_layers + 1, self.n_wb,))
        for j in range(self.n_layers + 1):
            for i in range(self.n_wb):
                flux = 0
                for k in range(self.n_streams):
                    flux += abs(self.incoming_radiation.diffuse[i] / pi * self.bvp_solutions[i, 1, k, j] * (
                            self.mus[k + 1] - self.mus[k]))

                actinic_flux_from_dif[j, i] = flux
        self.extras['actinic_flux_from_dif'] = abs(actinic_flux_from_dif)

        actinic_flux_from_dir = np.zeros((self.n_layers + 1, self.n_wb,))
        for j in range(self.n_layers + 1):
            for i in range(self.n_wb):
                flux = 0

                for k in range(self.n_streams):
                    flux += abs(self.incoming_radiation.direct[i] * self.bvp_solutions[i, 0, k, j] * (
                            self.mus[k + 1] - self.mus[k]))

                flux += self.incoming_radiation.direct[i] * self.direct_beam[j] / np.cos(self.incoming_radiation.sza)
                actinic_flux_from_dir[j, i] = flux
        self.extras['actinic_flux_from_dir'] = abs(actinic_flux_from_dir)

        self.extras['actinic_flux'] = abs(actinic_flux_from_dif + actinic_flux_from_dir)

        '''
        # Actinic flux
        # Dimension are [layer, wb]
        self.actinic_flux = np.zeros((self.n_layers + 1, self.n_wb,))

        for j in range(self.n_layers + 1):
            for i in range(self.n_wb):
                flux = 0

                for k in range(self.n_streams):
                    flux += abs(self.incoming_radiation.direct[i] * self.bvp_solutions[i, 0, k, j] * (
                            self.mus[k + 1] - self.mus[k]))
                    flux += abs(self.incoming_radiation.diffuse[i] / pi * self.bvp_solutions[i, 1, k, j] * (
                            self.mus[k + 1] - self.mus[k]))

                flux += self.incoming_radiation.direct[i] * self.direct_beam[j] / np.cos(
                    self.incoming_radiation.sza)
                self.actinic_flux[j, i] = flux
        '''

    def to_xr(self):
        # model output
        ds1 = xr.Dataset(data_vars={"down_flux_from_dif": (["z", "wl"], self.extras['down_flux_from_dif']),
                                    "up_flux_from_dif": (["z", "wl"], self.extras['up_flux_from_dif']),
                                    "net_flux_from_dif": (["z", "wl"], self.extras['net_flux_from_dif']),
                                    "down_flux_from_dir": (["z", "wl"], self.extras['down_flux_from_dir']),
                                    "up_flux_from_dir": (["z", "wl"], self.extras['up_flux_from_dir']),
                                    "net_flux_from_dir": (["z", "wl"], self.extras['net_flux_from_dir']),
                                    "net_flux": (["z", "wl"], self.extras['net_flux']),
                                    "actinic_flux_from_dif": (["z", "wl"], self.extras['actinic_flux_from_dif']),
                                    "actinic_flux_from_dir": (["z", "wl"], self.extras['actinic_flux_from_dir']),
                                    "actinic_flux": (["z", "wl"], self.extras['actinic_flux'])},
                         coords={"wl": ("wl", self.wb_centers),
                                 "z": ("z", self.canopy.z)})

        # model input
        ds2 = xr.Dataset(data_vars={"sza": ([], self.incoming_radiation.sza),
                                    "soil_r": (['wl'], self.canopy.soil_r)},
                         coords={"wl": ("wl", self.wb_centers)})

        # for each canopy component
        ds_comps = []
        for comp in self.canopy.components:
            label = comp.label

            # run some quick computations to get numbers to parameterize orientation, phase function
            mla, std, skew = Orient.mla_std_skew_top_half(comp.g_orient)
            r_forward_scatter_fraction, r_strong_bilambertial = Phase.quantify_forward_scatter(comp.r_phase)
            t_forward_scatter_fraction, t_strong_bilambertial = Phase.quantify_forward_scatter(comp.t_phase)

            ds_comp = xr.Dataset(data_vars={label + '_vertical_profile': (['z'], comp.c_profile),
                                            label + '_r': (['wl'], comp.r),
                                            label + '_t': (['wl'], comp.t),
                                            label + '_r_forward_scatter_fraction': ([], r_forward_scatter_fraction),
                                            label + '_r_strong_bilambertial': ([], r_strong_bilambertial),
                                            label + '_t_forward_scatter_fraction': ([], t_forward_scatter_fraction),
                                            label + '_t_strong_bilambertial': ([], t_strong_bilambertial),
                                            label + '_orient_mean': ([], mla),
                                            label + '_orient_std': ([], std),
                                            label + '_orient_skew': ([], skew)},
                                 coords={"wl": ("wl", self.wb_centers),
                                         "z": ("z", self.canopy.z)})
            ds_comps.append(ds_comp)

        ds = xr.merge([ds1, ds2] + ds_comps)
        return ds
