import numpy as np 
from scipy import signal
from scipy import misc

from dataloader import *
from guass import *
from visualization import *

# eq.(1)
def traditional_gaussian_1d():
    data = generate_1d_sine()
    kernel = guass1d(points=data.shape[0])
    mean_line = signal.convolve(data, kernel, mode='same', method='direct')

    show_lines([data, mean_line])

# eq.(3)
def traditional_gaussian_2d():
    data = load_measured_data()
    # kernel = guass2d()
    kernel = guass2d_sp()
    mean_surface = signal.convolve2d(data, kernel, boundary='fill', mode='same')

    show3d_surface(data)
    show3d_surface(mean_surface)

def traditional_gaussian_2d_two_steps():
    data = load_measured_data()
    kernel = guass1d()
    mean_surface = np.zeros(data.shape)

    # row based 1d convolve
    for y in range(data.shape[0]):
        mean_surface[y,:] = signal.convolve(data[y, :], kernel, mode='same', method='direct')
    # column base 1d convolve
    for x in range(data.shape[1]):
        mean_surface[:, x] = signal.convolve(mean_surface[:, x], kernel, mode='same', method='direct')
    show3d_surface(mean_surface)

 
if __name__ == "__main__":
    traditional_gaussian_1d()
    traditional_gaussian_2d()
    traditional_gaussian_2d_two_steps()

    waitkey()
    