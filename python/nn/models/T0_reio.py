import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import h5py as h5

import pytorch_spline

from .model import Model
from . import common
from .. import utils
from .. import time_slicing

class BasisDecompositionNet(nn.Module):
    def __init__(self, k, n_inputs_cosmo, n_inputs_tau, n_k):
        super().__init__()

        n_approx = 1

        self.k = k
        self.logk = nn.Parameter(torch.log(self.k), requires_grad=False)

        self.logk_spline = nn.Parameter(
                torch.log(utils.powerspace(self.k[0], self.k[-1], 5, 40)).float(),
                requires_grad=False
                )

        self.lin_cosmo_coeff = nn.Linear(n_inputs_cosmo, 20)
        self.lin_tau_coeff = nn.Linear(n_inputs_tau, 200)

        self.lin_coeff_1 = nn.Linear(20 + 200, 100)
        self.lin_coeff_2 = nn.Linear(100, len(self.logk_spline))

        self.relu = nn.LeakyReLU(0.3)

    def forward(self, x):
        cosmo = common.get_inputs_cosmo(x)
        tau_g = common.get_inputs_tau_reio(x)

        # approx_reio = x["t0_reio_approx_1"]

        y_cosmo = self.relu(self.lin_cosmo_coeff(cosmo))
        y_tau = self.relu(self.lin_tau_coeff(tau_g))

        y_spline = self.relu(self.lin_coeff_1(torch.cat((y_cosmo, y_tau), 1)))
        y_spline = self.lin_coeff_2(y_spline)

        return pytorch_spline.CubicSpline(self.logk_spline, y_spline)(self.logk)

class CorrectionNet(nn.Module):

    def __init__(self, n_inputs_cosmo, n_inputs_tau, n_k):
        super().__init__()

        self.lin_cosmo_corr = nn.Linear(n_inputs_cosmo, 200)
        self.lin_tau_corr = nn.Linear(n_inputs_tau, 200)

        self.lin_corr_1 = nn.Linear(200 + 200, 100)
        self.lin_corr_2 = nn.Linear(100, 200)
        self.lin_corr_3 = nn.Linear(200, n_k)

        self.relu = nn.LeakyReLU(0.3)

    def forward(self, x):
        cosmo = common.get_inputs_cosmo(x)
        tau_g = common.get_inputs_tau_reio(x)

        y_cosmo = self.relu(self.lin_cosmo_corr(cosmo))
        y_tau = self.relu(self.lin_tau_corr(tau_g))

        y = torch.cat((y_cosmo, y_tau), axis=1)
        y = self.relu(self.lin_corr_1(y))
        y = self.relu(self.lin_corr_2(y))
        y = self.lin_corr_3(y)

        return y

class Net_ST0_Reio(Model):

    def __init__(self, k):
        super().__init__(k)

        n_inputs_cosmo = len(common.INPUTS_COSMO)
        n_inputs_tau = 4
        n_k = len(k)

        self.lin_cosmo = nn.Linear(n_inputs_cosmo, 20)
        self.lin_tau = nn.Linear(n_inputs_tau, 133)

        self.lin_combined = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.lin_cosmo.out_features + self.lin_tau.out_features, 500),
            nn.PReLU(),
            nn.Linear(500, n_k)
        )
    def forward(self, x):
        self.k_min = x["k_min"][0]

        inputs_cosmo = common.get_inputs_cosmo(x)
        inputs_tau = torch.stack([
            x["tau_relative_to_reio"],
            x["g_reio"],
            x["g_reio_prime"],
            x["e_kappa"],
        ], axis=1)

        prediction = self.lin_combined(
            torch.cat((
                self.lin_cosmo(inputs_cosmo),
                self.lin_tau(inputs_tau)
            ), dim=1)
        )
        return prediction

    def epochs(self):
        return 40

    def slicing(self):
        return time_slicing.TimeSlicingReio(0.6)

    def optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def lr_scheduler(self, opt):
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch: np.exp(-epoch / 8))

    def required_inputs(self):
        return set(common.INPUTS_COSMO + [
            "k_min",
            "tau_relative_to_reio",
            "e_kappa",
            "g_reio", "g_reio_prime",
            # "t0_reio_approx_1",
            ])

    def tau_training(self):
        with h5.File(os.path.join(os.path.expandvars("$CLASSNET_DATA"), "tau_t0_reio.h5"), "r") as f:
            tau_training = f["tau"][()]
        return tau_training

    def source_functions(self):
        return ["t0_reio_no_isw"]

    def criterion(self):
        def loss(prediction, truth):
            return common.mse_truncate(self.k, self.k_min)(prediction, truth)
        return loss

if __name__ == "__main__":
    iface = interface.TrainingInterface(Net_ST0_Reio)
    iface.run()
