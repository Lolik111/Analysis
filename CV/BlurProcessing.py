import cv2
import numpy as np
import scipy
from skimage import restoration

def modifiedLaplacian(img):
    ''''LAPM' algorithm (Nayar89)'''
    M = np.array([-1, 2, -1])
    G = np.array([0.25, 0.5, 0.25])#cv2.getGaussianKernel(ksize=3, sigma=-1)
    Lx = cv2.sepFilter2D(src=img, ddepth=cv2.CV_64F, kernelX=M, kernelY=M)
    Ly = cv2.sepFilter2D(src=img, ddepth=cv2.CV_64F, kernelX=G, kernelY=G)
    FM = np.abs(Lx) + np.abs(Ly)
    return cv2.mean(FM)[0]



def varianceOfLaplacian(img):
    ''''LAPV' algorithm (Pech2000)'''
    lap = cv2.Laplacian(img, ddepth=-1)#cv2.cv.CV_64F)
    stdev = cv2.meanStdDev(lap)[1]
    s = stdev[0]**2
    return s[0]



def tenengrad(img, ksize=3):
    ''''TENG' algorithm (Krotkov86)'''
    Gx = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize)
    Gy = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize)
    FM = Gx**2 + Gy**2
    return cv2.mean(FM)[0]



def normalizedGraylevelVariance(img):
    ''''GLVN' algorithm (Santos97)'''
    mean, stdev = cv2.meanStdDev(img)
    s = stdev[0]**2 / mean[0]
    return s[0]


def Laplacian(img, ksize):
    fm = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)
    return fm.var()


def strangeAlg(img):
    h_matrix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    v_matrix = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    sum_squares = cv2.norm(h_matrix) * cv2.norm(h_matrix) + cv2.norm(v_matrix) * cv2.norm(v_matrix)
    metric = 1. / (sum_squares / (img.shape[0] * img.shape[1]))
    return metric


def contrast_measure(img, ksize=3):
    Gx = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize)
    Gy = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize)
    res = cv2.magnitude(Gx, Gy)
    return sum(res)[0]


def random_metric(img):
    height, width = img.shape[:2]

    sum = 0
    for x in range(width - 1):
        for y in range(height):
            sum += abs(int(img[y, x]) - int(img[y, x+1]))
    return sum


def another_lap(img):
    return np.max(cv2.convertScaleAbs(cv2.Laplacian(img, 3)))


def canny_method(img):
    edges = cv2.Canny(img, 100, 200)
    return np.count_nonzero(edges)


def canny_area_dependent_method(img):
    edges = cv2.Canny(img, 100, 200)
    return np.count_nonzero(edges) * 1000. / (edges.shape[0] * edges.shape[1])



def detect_exposion_level(img):
    histr = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256])
    return np.sum(histr[0:210]) / (np.sum(histr[210:-1]) + 1)
