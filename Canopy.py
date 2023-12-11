import numpy as np
import matplotlib.pyplot as plt
import CanopyComponent as CC


def calc_d_profile(c_profile):
    d_profile = np.zeros(len(c_profile)-1)

    for i in range(len(d_profile)):
        d_profile[i] = c_profile[i+1] - c_profile[i]

    return d_profile


def calc_z_centers(z):
    z_centers = np.zeros(len(z) - 1)

    for i in range(len(z_centers)):
        z_centers[i] = (z[i] + z[i+1]) / 2

    return z_centers


class Canopy:

    def __init__(self, z_profile, n_wb):
        self.z = np.array(z_profile)
        self.dz = calc_d_profile(self.z)
        self.z_centers = calc_z_centers(self.z)
        self.n_layers = len(self.z) - 1
        self.n_wb = n_wb
        self.components = []
        self.soil_r = None

    def add_canopy_component(self, canopy_component):
        assert type(canopy_component) is CC.CanopyComponent, "Object must be type CanopyComponent"

        # appropriate number of layers
        assert len(canopy_component.c_profile) == len(self.z), "Ensure proper number of layers in CanopyComponent"
        assert len(canopy_component.d_profile) == len(self.dz), "Ensure proper number of layers in CanopyComponent"

        # optical properties at all wavelengths
        assert len(canopy_component.r) == self.n_wb, "Ensure correct number of reflectance values"
        assert len(canopy_component.t) == self.n_wb, "Ensure correct number of transmittance values"
        assert len(canopy_component.a) == self.n_wb, "Ensure correct number of reflectance and transmittance values"

        self.components.append(canopy_component)

    def remove_canopy_component(self, label_to_remove):
        for i, component in enumerate(self.components):
            if component.label == label_to_remove:
                self.components.pop(i)

    def __str__(self):
        return self.get_string()

    def get_string(self):
        string = str(len(self.z)) + " layer canopy with " + str(len(self.components)) + " different components:\n"
        for comp in self.components:
            string += '  ' + str(comp.get_string()) + '\n'
        return string

    def get_canopy_z_profile(self):
        return self.z

    def get_canopy_components(self):
        return self.components

    def set_soil(self, soil_r):
        soil_r = np.array(soil_r)

        assert len(soil_r) == self.n_wb, "Ensure wavebands match for soil reflectance"
        for r in soil_r:
            assert 0 <= r <= 1, "Soil reflectance must be between 0 and 1"

        self.soil_r = soil_r




