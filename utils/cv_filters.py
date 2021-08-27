import bisect
import numpy as np
import scipy.ndimage as nd
from numpy.fft import fft2, ifft2
from scipy import signal


def bresenham(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))

    return points


def strel_line(length, degrees):
    if length >= 1:
        theta = degrees * np.pi / 180
        x = round((length - 1) / 2 * np.cos(theta))
        y = -round((length - 1) / 2 * np.sin(theta))
        points = bresenham(-x, -y, x, y)
        points_x = [point[0] for point in points]
        points_y = [point[1] for point in points]
        n_rows = int(2 * max([abs(point_y) for point_y in points_y]) + 1)
        n_columns = int(2 * max([abs(point_x) for point_x in points_x]) + 1)
        strel = np.zeros((n_rows, n_columns))
        rows = ([point_y + max([abs(point_y) for point_y in points_y]) for point_y in points_y])
        columns = ([point_x + max([abs(point_x) for point_x in points_x]) for point_x in points_x])
        idx = []
        for x in zip(rows, columns):
            idx.append(np.ravel_multi_index((int(x[0]), int(x[1])), (n_rows, n_columns)))
        strel.reshape(-1)[idx] = 1
    return strel


def imadjust(src, tol=1, vin=[0, 255], vout=(0, 255)):
    dst = src.copy()
    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.zeros(256, dtype=np.int)
        for r in range(src.shape[0]):
            for c in range(src.shape[1]):
                hist[src[r, c]] += 1
        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, len(hist)):
            cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    for r in range(dst.shape[0]):
        for c in range(dst.shape[1]):
            vs = max(src[r, c] - vin[0], 0)
            vd = min(int(vs * scale + 0.5) + vout[0], vout[1])
            dst[r, c] = vd
    return dst


def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = fft2(dummy)
    kernel = fft2(kernel, s=img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(ifft2(dummy))
    return dummy


def gaussian_kernel(kernel_size=3):
    h = signal.gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = np.dot(h, h.transpose())
    h /= np.sum(h)
    return h


def laplacianOfGaussian(img, par):
    LoG = nd.gaussian_laplace(img, 2)
    thres = np.absolute(LoG).mean() * par
    output = np.zeros(LoG.shape)
    w = output.shape[1]
    h = output.shape[0]
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            patch = LoG[y - 1:y + 2, x - 1:x + 2]
            p = LoG[y, x]
            maxP = patch.max()
            minP = patch.min()
            if p > 0:
                zeroCross = True if minP < 0 else False
            else:
                zeroCross = True if maxP > 0 else False
            if ((maxP - minP) > thres) and zeroCross:
                output[y, x] = 1
    return output