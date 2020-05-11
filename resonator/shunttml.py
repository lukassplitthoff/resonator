"""
This module contains models and fitters for resonators that are operated in the transmission line with variable load configuration.
"""
from __future__ import absolute_import, division, print_function

import numpy as np

from . import background, base, guess, linear, kerr


class AbstractShunt(base.ResonatorModel):
    """
    This is an abstract class that models a resonator operated in the shunt-coupled configuration.
    """
    # This is the value of the scattering data far from resonance.
    reference_point = 1 + 0j


    # See kerr.kerr_detuning_shift for the meaning of this.
    io_coupling_coefficient = 1 / 2


# Linear models and fitters

class TLResonator(AbstractShunt):
    """
    This class models a linear transmission line resonator ...
    """

    def __init__(self, *args, **kwds):
        """
        This class can be used directly, like any lmfit Model, but it is easier to use the LinearShuntFitter wrapper
        class that is defined in this module.

        :param args: arguments passed directly to lmfit.model.Model.__init__().
        :param kwds: keywords passed directly to lmfit.model.Model.__init__().
        """

        def transmissionline_loaded(frequency, Rl, Cc, Ls, Rs, Cs, Gs, imp_load, length):
            omega = 2 * np.pi * frequency
            imp_coupling = Rl/2 + 1/(1j*omega*Cc)
            imp_char = np.sqrt((Rs + 1j * omega * Ls) / (Gs + 1j * omega * Cs))
            prop_const = np.sqrt((Rs + 1j * omega * Ls) * (Gs + 1j * omega * Cs))
            imp_line = imp_char * (imp_load + imp_char * np.tanh(prop_const * length)) / (imp_char + imp_load * np.tanh(prop_const * length))
            imp = imp_coupling + imp_line

            return imp

        super(TLResonator, self).__init__(func=transmissionline_loaded, *args, **kwds)

    def guess(self, data=None, frequency=None, **kwds):
        params = self.make_params()
        params['Rl'].set(value=50, min=0, max=100)
        params['Cc'].set(value=10e-12, min=0.5e-12, max=100e-12)
        params['Ls'].set(value=3e-9, min=0.3e-9, max=30e-9)
        params['Rs'].set(value=1)
        params['Cs'].set(value=0.4e-12, min=0.04e-12, max=40e-12)
        params['Gs'].set(value=1)
        params['imp_load'].set(value=0, vary=False)
        params['length'].set(value=4000e-6, min=2525e-6, max=6755e-6)
        return params


class TLResonatorFitter(linear.LinearResonatorFitter):
    """
    This class fits data from a linear shunt-coupled resonator.

    Here, linear means that the energy stored in the resonator is proportional to the power on the feedline. When the
    input power is sufficiently high, this relationship will start to become more complicated and the shape of the
    resonance will change. In this case, the resonance can still be fit with a Kerr model.
    """

    def __init__(self, frequency, data, background_model=None, errors=None, **kwds):
        """
        Fit the given data to a composite model that is the product of a background model and the LinearShunt model.

        :param frequency: an array of floats containing the frequencies at which the data was measured.
        :param data: an array of complex numbers containing the data.
        :param background_model: an instance (not the class) of a model representing the background response without the
          resonator; the default of `background.MagnitudePhase()` assumes that this is modeled well by a single complex
          constant at all frequencies.
        :param errors: an array of complex numbers containing the standard errors of the mean of the data points.
        :param kwds: keyword arguments passed directly to `lmfit.model.Model.fit()`.
        """
        if background_model is None:
            background_model = background.MagnitudePhase()
        super(TLResonatorFitter, self).__init__(frequency=frequency, data=data, foreground_model=TLResonator(),
                                                background_model=background_model, errors=errors, **kwds)

    # ToDo: math
    def invert(self, scattering_data):
        pass


