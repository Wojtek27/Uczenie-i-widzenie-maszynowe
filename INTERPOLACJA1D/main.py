import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt

# liczba punktów interpolacji
n = 10

# przedział interpolacji
x = np.linspace(np.e**-8, 2*np.pi, n+1)

# funkcje do interpolacji
y1 = np.sin(x**(-1))
y2 = np.sign(np.sin(8*x))
y3 = np.sin(x)

# interpolacja funkcji sin(x**(-1))
cs1_0 = CubicSpline(x, y1, bc_type='natural')
cs1_1 = CubicSpline(x, y1, bc_type='clamped')
cs1_2 = CubicSpline(x, y1, bc_type='not-a-knot')

# interpolacja funkcji signum(sin(8*x))
cs2_0 = CubicSpline(x, y2, bc_type='natural')
cs2_1 = CubicSpline(x, y2, bc_type='clamped')
cs2_2 = CubicSpline(x, y2, bc_type='not-a-knot')

# interpolacja funkcji sin(x)
cs3_0 = CubicSpline(x, y3, bc_type='natural')
cs3_1 = CubicSpline(x, y3, bc_type='clamped')
cs3_2 = CubicSpline(x, y3, bc_type='not-a-knot')

# generowanie wektora x dla wykresu
x_plot = np.linspace(0, 2*np.pi, 1000)

# wykres funkcji sin(x**(-1))
plt.figure()
plt.plot(x, y1, 'o', label='f(x)')
plt.plot(x_plot, cs1_0(x_plot), label='B0')
plt.plot(x_plot, cs1_1(x_plot), label='B1')
plt.plot(x_plot, cs1_2(x_plot), label='B2')
plt.legend()
plt.title('Interpolacja sin(x**(-1))')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

# wykres funkcji signum(sin(8*x))
plt.figure()
plt.plot(x, y2, 'o', label='f(x)')
plt.plot(x_plot, cs2_0(x_plot), label='B0')
plt.plot(x_plot, cs2_1(x_plot), label='B1')
plt.plot(x_plot, cs2_2(x_plot), label='B2')
plt.legend()
plt.title('Interpolacja signum(sin(8*x))')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

# wykres funkcji sin(x)
plt.figure()
plt.plot(x, y3, 'o', label='f(x)')
plt.plot(x_plot, cs3_0(x_plot), label='B0')
plt.plot(x_plot, cs3_1(x_plot), label='B1')
plt.plot(x_plot, cs3_2(x_plot), label='B2')
plt.legend()
plt.title('Interpolacja sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

# interpolacja funkcji sin(x**(-1))
ks1 = InterpolatedUnivariateSpline(x, y1, k=2)

# interpolacja funkcji signum(sin(8*x))
ks2 = InterpolatedUnivariateSpline(x, y2, k=2)

# interpolacja funkcji sin(x)
ks3 = InterpolatedUnivariateSpline(x, y3, k=2)

# wykres funkcji sin(x**(-1))
plt.figure()
plt.plot(x, y1, 'o', label='f(x)')
plt.plot(x_plot, ks1(x_plot), label='Keys')
plt.legend()
plt.title('Interpolacja sin(x**(-1))')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

# wykres funkcji signum(sin(8*x))
plt.figure()
plt.plot(x, y2, 'o', label='f(x)')
plt.plot(x_plot, ks2(x_plot), label='Keys')
plt.legend()
plt.title('Interpolacja signum(sin(8*x))')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

# wykres funkcji signum(sin(8*x))
plt.figure()
plt.plot(x, y3, 'o', label='f(x)')
plt.plot(x_plot, ks3(x_plot), label='Keys')
plt.legend()
plt.title('Interpolacja sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

plt.show()


# funkcja obcietej sinc
def sinc(x):
    return np.sinc(x/np.pi) * (abs(x) < np.pi)

a = 0
b = 2*np.pi
# obciecie funkcji sinc
y1_sinc = y1 * sinc(x-a) * sinc(b-x)
y2_sinc = y2 * sinc(x-a) * sinc(b-x)
y3_sinc = y3 * sinc(x-a) * sinc(b-x)

# interpolacja funkcji
cs1 = CubicSpline(x, y1_sinc)
cs2 = CubicSpline(x, y2_sinc)
cs3 = CubicSpline(x, y3_sinc)

# przedzial, na którym rysujemy funkcje interpolacyjne
x_plot = np.linspace(a, b, 10*n)

# rysowanie funkcji interpolacyjnych
plt.plot(x_plot, cs1(x_plot), label='sin(x**(-1))')
plt.plot(x_plot, cs2(x_plot), label='signum(sin(8*x))')
plt.plot(x_plot, cs3(x_plot), label='sin(x)')
plt.plot(x, y1, 'o', label='punkty interpolacji sin(x**(-1))')
plt.plot(x, y2, 'o', label='punkty interpolacji signum(sin(8*x))')
plt.plot(x, y3, 'o', label='punkty interpolacji sin(x)')
plt.legend()
plt.grid(visible=True)
plt.show()