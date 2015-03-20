from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist # distance between realspace points
from scipy import sparse


class ChargeState:

	esys = False
	ne = 0

	def __init__(self, esys, ne):	
		self.esys = esys
		self.ne = ne



class SzSpinState:

	esys = False
	nu = 0
	nd = 0

	def __init__(self, esys, nu, nd):
		self.esys = esys
		self.nu = nu
		self.nd = nd
 

	@staticmethod
	def state():
		return True

