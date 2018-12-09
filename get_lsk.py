# coding： utf-8

__author__ = 'Cclock'

import cv2
import math
import numpy
from scipy import signal


_epsa = _opzero = 0.0000001


def gradient(matrix):

    xfilter = numpy.array([[-0.5, 0, 0.5]])
    yfilter = numpy.array([[-0.5], [0], [0.5]])
    xfilter = numpy.rot90(xfilter, 2)
    yfilter = numpy.rot90(yfilter, 2)
    gx = signal.convolve2d(matrix, xfilter, boundary='symm', mode='same')
    gy = signal.convolve2d(matrix, yfilter, boundary='symm', mode='same')

    return gx, gy


def gen_gaussian(winsize):

    sigma = 0.8
    winsize = math.floor(winsize / 2)
    y, x = numpy.ogrid[-winsize:winsize+1, -winsize:winsize+1]
    disk = numpy.exp( -(x**2 + y**2)**1.7 / (2.*sigma**2) )

    return disk.astype(float)


def edge_mirror(matrix, winsize):

    rows, cols = matrix.shape
    winsize = math.floor(winsize / 2) + 1
    left = matrix[:, 1:winsize][:, ::-1]
    right = matrix[:, cols-winsize:cols-1][:, ::-1]
    matrix = numpy.concatenate([left, matrix, right], axis=1)
    top = matrix[1:winsize, :][::-1, :]
    buttom = matrix[rows-winsize:rows-1, :][::-1, :]
    matrix = numpy.concatenate([top, matrix, buttom], axis=0)

    return matrix


def cov_matrix(gx, gy, winsize, alpha):

    """
    gradients covariance matrix
    | gx^2 gx*gy |
    | gx*gy gy^2 |

    """

    gx = edge_mirror(gx, winsize)
    gy = edge_mirror(gy, winsize)
    radius_filter = gen_gaussian(winsize)
    radius_filter = numpy.rot90(radius_filter, 2)

    lenth = sum(sum(radius_filter))

    gx = signal.convolve2d(gx, radius_filter, mode='valid')
    gy = signal.convolve2d(gy, radius_filter, mode='valid')

    c11 = numpy.multiply(gx, gx)
    c22 = numpy.multiply(gy, gy)
    c12 = numpy.multiply(gx, gy)


    # SVD closed form
    lambda1 = (c11 + c22 + numpy.sqrt((c11 - c22)**2 + 4*c12**2)) / 2
    lambda2 = (c11 + c22 - numpy.sqrt((c11 - c22)**2 + 4*c12**2)) / 2
    numer = c11 + c12 - lambda1
    denom = c22 + c12 - lambda2

    ev1 = numpy.zeros_like(numer)
    ev2 = numpy.zeros_like(ev1)

    rows, cols = numer.shape
    for r in range(rows):
        for c in range(cols):
            if abs(denom[r, c]) < _opzero:
                if abs(numer[r, c]) < _opzero:
                    if abs(denom[r, c]) > abs(numer[r, c]):
                        ev1[r, c] = 0
                        ev2[r, c] = 1
                    else:
                        ev1[r, c] = 1
                        ev2[r, c] = 0
                else:
                    ev1[r, c] = 1
                    ev2[r, c] = 0
            else:
                theta = math.atan(-numer[r, c]/denom[r, c])
                ev1 = math.sin(theta)
                ev2 = math.cos(theta)

            sv1 = math.sqrt(abs(lambda1[r, c]))
            sv2 = math.sqrt(abs(lambda2[r, c]))
            p = ((sv1 * sv2 + _epsa) / lenth)**alpha
            s1 = (sv1 + 1) / (sv2 + 1)
            s2 = 1. / s1
            c11[r, c] = p * (s1 * ev2 ** 2 + s2 * ev1 ** 2)
            c22[r, c] = p * (s1 * ev1 ** 2 + s2 * ev2 ** 2)
            c12[r, c] = p * (s1 - s2) * ev1 * ev2

    c11 = edge_mirror(c11, winsize)
    c12 = edge_mirror(c12, winsize)
    c22 = edge_mirror(c22, winsize)

    return c11, c12, c22


def lark(cov_matirx):





if __name__ == '__main__':
    img = numpy.array([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])
    gx, gy = gradient(img)
    c11, c12, c22 = cov_matrix(gx, gy, 3, 0.4)
