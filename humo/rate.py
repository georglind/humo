from __future__ import division, print_function

import numpy as np
import warnings


def steady_state_occupations(rates):
    """
    Find the occupations from a given set of rates
    """
    rateq = rates - np.diag(np.sum(rates, axis=0))
    p = np.linalg.null(rateq)
    return p
