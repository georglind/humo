from __future__ import division, print_function
import numpy as np
import sys
import time
from IPython import display

def message(strs, verbal):
    if verbal:
        print(strs)


def choose(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
    """
    n = int(n)
    k = int(k)

    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in xrange(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


class Progress:
    """
    Creates a simple inline progress bar for receiving updates.
    """
    def __init__(self, steps):
        self.barsteps = 60
        self.barindex = 0
        self.steps = int(steps)
        self.index = 0

        print('Progress:')
        self.update("", "")

    def update(self, ending="", starting="\r"):
        sys.stdout.write(starting)
        display.clear_output(wait=True)
        print('[', end="")
        [print('-', end="") for i in xrange(self.barindex)]
        [print(' ', end="") for i in xrange(self.barsteps-self.barindex)]
        print('] ', end="")
        print('{0}%'.format(int(100*self.index/self.steps)), end=ending)
        time.sleep(.01)

    def step(self, step=1):
        if self.index >= self.steps:  # the bar has been filled
            return

        self.index += step

        if self.index == self.steps:  # this is the last increment. Fill remaining bar.
            self.barindex = self.barsteps
            self.update("\n")
            return
            # return True

        self.barindex = int(np.floor(self.barsteps/self.steps*self.index))
        self.update()

        # return True
