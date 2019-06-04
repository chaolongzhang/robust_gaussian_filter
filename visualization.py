import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

plt.ion()

def show_line(data):
    plt.figure()
    plt.plot(data)
    plt.show()

def show_lines(lines):
    plt.figure()
    for line in lines:
        plt.plot(line)
    plt.show()

def show3d_surface(data):
    rows = len(data)
    cols = len(data[0])
    z = np.array(data)
    x = np.arange(cols)
    y = np.arange(rows)
    x, y = np.meshgrid(x, y)

    fig = plt.figure()
    plt.title('surface')
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(x, y, z, rstride=5, cstride=5, cmap=cm.jet, linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.2, aspect=1)

    plt.show()

def waitkey():
    input("Press Enter to continue...")
