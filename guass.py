import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

LAMBDA = 0.5

def guass1d(points = 1000):
    x = np.linspace(-1, 1, points)
    alpha = np.sqrt(np.log(2)/np.pi)
    flambda = LAMBDA
    y = (1 / (alpha * flambda)) * np.exp(-np.pi * (x / (alpha * flambda)) ** 2)
    y = y / np.sum(y)
    return y

def guass2d(xPoints = 100, yPoints = 100):
    x = np.linspace(-1, 1, xPoints)
    y = np.linspace(-1, 1, yPoints)
    alpha = np.sqrt(np.log(2)/np.pi)
    lambdaX = LAMBDA
    lambdaY = LAMBDA
    z = np.zeros([yPoints, xPoints])

    for yi in range(len(y)):
        z[yi,:] =  (1 / (alpha ** 2 * lambdaX * lambdaY)) * np.exp(-np.pi * (y[yi] / (alpha * lambdaX)) ** 2 - np.pi * (x / (alpha * lambdaY)) ** 2)
    
    z = z / np.sum(z)
    return z

def guass2d_sp(xPoints = 100, yPoints = 100):
    x = np.linspace(-1, 1, xPoints)
    y = np.linspace(-1, 1, yPoints)
    alpha = np.sqrt(np.log(2)/np.pi)
    lambdaX = LAMBDA
    lambdaY = LAMBDA
    sx = (1 / (alpha * lambdaX)) * np.exp(-np.pi * (x / (alpha * lambdaX)) ** 2)
    sy = (1 / (alpha * lambdaY)) * np.exp(-np.pi * (y / (alpha * lambdaY)) ** 2)
    z = np.zeros([yPoints, xPoints])

    for yi in range(len(y)):
        for xi in range(len(x)):
            z[yi, xi] = sx[yi] * sy[xi]

    z = z / np.sum(z)
    return z


if __name__ == "__main__":
    from visualization import *
    data1 = guass1d(3)
    data2 = guass2d()
    data3 = guass2d_sp()

    show_line(data1)
    show3d_surface(data2)
    show3d_surface(data3)

    waitkey()
