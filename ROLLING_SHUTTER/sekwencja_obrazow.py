import numpy as np
import matplotlib.pyplot as plt

M = 64
n = 3

pi = np.pi
sin = np.sin
m = M/2
x = np.linspace(-pi, pi)

for i in range(6):
    f = sin(n*x +(m*pi/100))
    plt.polar(x + i, f)

plt.show()

