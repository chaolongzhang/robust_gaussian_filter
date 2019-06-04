import numpy as np 
from scipy import signal
from scipy import linalg
from statsmodels import robust
import logging

from run_time import Runtime
from guass import *
from dataloader import *
from visualization import *


FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)


class RobustAralGaussianRegression(object):

    def __init__(self, measured_data):
        super(RobustAralGaussianRegression).__init__()
        self.STOP = 100
        self.measured_data = measured_data
        Y, X = measured_data.shape
        self.s = guass2d(X, Y)
        self.sx = guass1d(X).reshape(1, X)
        self.sy = guass1d(Y).reshape(Y, 1)

    def init_delta(self):
        self.delta = np.ones_like(self.measured_data)

    def update_delta(self, c):
        idx = self.w > c
        self.delta = (1 - (self.w / c) ** 2) ** 2
        self.delta[idx] = 0

    # eq.17
    def caculateAij(self, i, j):
        logging.debug('start caculateAij...')
        X, Y = self.delta.shape
        A = np.zeros_like(self.delta)
        for x in range(X):
            for y in range(Y):
                kernel = ((x + 1) ** i) * ((y + 1) ** j) * self.s
                temp = signal.convolve2d(self.delta, kernel, mode='valid')[0]
                A[x, y] = temp
        logging.debug('finished caculateAij...')
        return A

    # eq.19
    async def caculateAij_fast(self, i, j):
        logging.debug('start caculateAij_fast...')
        X, Y = self.delta.shape
        A = np.zeros_like(self.delta)
        data = self.delta
        for x in range(X):
            for y in range(Y):
                row_data = signal.convolve2d(data, (x + 1) ** i * self.sx, mode='valid')
                temp = signal.convolve2d(row_data, (y + 1) ** j * self.sy, mode='valid')
                A[x, y] = temp[0, 0]
        logging.debug('finished caculateAij_fast...')
        return A

    def caculateA(self):
        logging.debug('start caculateA...')
        self.A00 = self.caculateAij(0, 0)
        self.A10 = self.caculateAij(1, 0)
        self.A01 = self.caculateAij(0, 1)
        self.A20 = self.caculateAij(2, 0)
        self.A11 = self.caculateAij(1, 1)
        self.A02 = self.caculateAij(0, 2)
        self.A30 = self.caculateAij(3, 0)
        self.A21 = self.caculateAij(2, 1)
        self.A12 = self.caculateAij(1, 2)
        self.A03 = self.caculateAij(0, 3)
        self.A40 = self.caculateAij(4, 0)
        self.A31 = self.caculateAij(3, 1)
        self.A22 = self.caculateAij(2, 2)
        self.A13 = self.caculateAij(1, 3)
        self.A04 = self.caculateAij(0, 4)
        logging.debug('finished caculateA...')

    def caculateA_fast(self):
        logging.debug('start caculateA_fast...')
        self.A00 = self.caculateAij_fast(0, 0)
        self.A10 = self.caculateAij_fast(1, 0)
        self.A01 = self.caculateAij_fast(0, 1)
        self.A20 = self.caculateAij_fast(2, 0)
        self.A11 = self.caculateAij_fast(1, 1)
        self.A02 = self.caculateAij_fast(0, 2)
        self.A30 = self.caculateAij_fast(3, 0)
        self.A21 = self.caculateAij_fast(2, 1)
        self.A12 = self.caculateAij_fast(1, 2)
        self.A03 = self.caculateAij_fast(0, 3)
        self.A40 = self.caculateAij_fast(4, 0)
        self.A31 = self.caculateAij_fast(3, 1)
        self.A22 = self.caculateAij_fast(2, 2)
        self.A13 = self.caculateAij_fast(1, 3)
        self.A04 = self.caculateAij_fast(0, 4)
        logging.debug('finished caculateA...')

    # eq.17
    def caculateFij(self, i, j):
        logging.debug('start caculateFij...')
        X, Y = self.delta.shape
        F = np.zeros_like(self.delta)
        data = self.delta * self.measured_data
        for x in range(X):
            for y in range(Y):
                kernel = ((x + 1) ** i) * ((y + 1) ** j) * self.s
                temp = signal.convolve2d(data, kernel, mode='valid')[0]
                F[x, y] = temp
        logging.debug('finished caculateFij...')
        return F

    # eq.20
    def caculateFij_fast(self, i, j):
        logging.debug('start caculateFij_fast...')
        X, Y = self.delta.shape
        F = np.zeros_like(self.delta)
        data = self.delta * self.measured_data
        for x in range(X):
            for y in range(Y):
                row_data = signal.convolve2d(data, (x + 1) ** i * self.sx, mode='valid')
                temp = signal.convolve2d(row_data, (y + 1) ** j * self.sy, mode='valid')
                F[x, y] = temp[0, 0]
        logging.debug('finished caculateFij_fast...')
        return F

    def caculateF(self):
        self.F00 = self.caculateFij(0, 0)
        self.F10 = self.caculateFij(1, 0)
        self.F01 = self.caculateFij(0, 1)
        self.F20 = self.caculateFij(2, 0)
        self.F11 = self.caculateFij(1, 1)
        self.F02 = self.caculateFij(0, 2)

    def caculateF_fast(self):
        self.F00 = self.caculateFij_fast(0, 0)
        self.F10 = self.caculateFij_fast(1, 0)
        self.F01 = self.caculateFij_fast(0, 1)
        self.F20 = self.caculateFij_fast(2, 0)
        self.F11 = self.caculateFij_fast(1, 1)
        self.F02 = self.caculateFij_fast(0, 2)

    def solve_linear_equation(self):
        logging.debug('start combine matrix...')
        row1 = np.hstack((self.A00, self.A10, self.A01, self.A20, self.A11, self.A02))
        row2 = np.hstack((self.A10, self.A20, self.A11, self.A30, self.A21, self.A12))
        row3 = np.hstack((self.A01, self.A11, self.A02, self.A21, self.A12, self.A03))
        row4 = np.hstack((self.A20, self.A30, self.A21, self.A40, self.A31, self.A22))
        row5 = np.hstack((self.A11, self.A21, self.A12, self.A31, self.A22, self.A13))
        row6 = np.hstack((self.A02, self.A13, self.A03, self.A13, self.A13, self.A04))
        A = np.vstack((row1, row2, row3, row4, row5, row6))
        F = np.vstack((self.F00, self.F10, self.F01, self.F20, self.F11, self.F02))
        logging.debug('finished combine matrix...')

        logging.debug('start solve linear equation...')
        n, m = A.shape
        # if n == m:
            # X = linalg.solve(A, F)
        # else:
        X = linalg.lstsq(A, F)[0]
        self.w = X[0: self.measured_data.shape[0], :]
        logging.debug('finished solve linear equation...')

    def MAD(self):
        m = robust.mad(self.w)[0]
        logging.debug('MAD is %s', m)
        assert(m != 0)
        return m

    def run(self):
        moniter = Runtime()
        self.init_delta()
        Cb0= 100
        while True:
            moniter.start()
            
            self.caculateA_fast()
            self.caculateF_fast()
            self.solve_linear_equation()
            Cb1 = self.MAD()
            moniter.end()
            print('time: ', moniter.desc())

            d = abs((Cb0 - Cb1) / (Cb0 + 1E-10))
            print('difference between c:', d)
            if d > self.STOP:
                break

            waitkey()
            Cb0 = Cb1
            self.update_delta(Cb0)


def main():
    # measured_data = generate_2d()
    measured_data = load_measured_data()
    gr = RobustAralGaussianRegression(measured_data)
    gr.run()


if __name__ == "__main__":
    main()

