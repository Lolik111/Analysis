from matplotlib import pyplot as plt
import cv2
import numpy as np
import scipy
import math
from skimage import restoration, io, color, draw
from CV.LineDict import LineDictionary
from CV.BlindDeconvolution import recover_image, estimate_psf, unblur
import os

def modifiedLaplacian(img):
    ''''LAPM' algorithm (Nayar89)'''
    M = np.array([-1, 2, -1])
    G = cv2.getGaussianKernel(ksize=3, sigma=-1)
    Lx = cv2.sepFilter2D(src=img, ddepth=cv2.CV_64F, kernelX=M, kernelY=M)
    Ly = cv2.sepFilter2D(src=img, ddepth=cv2.CV_64F, kernelX=G, kernelY=G)
    FM = np.abs(Lx) + np.abs(Ly)
    return cv2.mean(FM)[0]


def varianceOfLaplacian(img):
    ''''LAPV' algorithm (Pech2000)'''
    lap = cv2.Laplacian(img, ddepth=-1)  # cv2.cv.CV_64F)
    stdev = cv2.meanStdDev(lap)[1]
    s = stdev[0] ** 2
    return s[0]


def tenengrad(img, ksize=3):
    ''''TENG' algorithm (Krotkov86)'''
    Gx = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize)
    Gy = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize)
    FM = Gx ** 2 + Gy ** 2
    return cv2.mean(FM)[0]


def normalizedGraylevelVariance(img):
    ''''GLVN' algorithm (Santos97)'''
    mean, stdev = cv2.meanStdDev(img)
    s = stdev[0] ** 2 / mean[0]
    return s[0]


def Laplacian(img, ksize):
    fm = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)
    return fm.var()


def strangeAlg(img):
    h_matrix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    v_matrix = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    sum_squares = cv2.norm(h_matrix) * cv2.norm(h_matrix) + cv2.norm(v_matrix) * cv2.norm(v_matrix)
    metric =  sum_squares / (img.shape[0] * img.shape[1])
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
            sum += abs(int(img[y, x]) - int(img[y, x + 1]))
    return sum


def another_lap(img):
    return np.max(cv2.convertScaleAbs(cv2.Laplacian(img, 3)))


def canny_method(img):
    edges = cv2.Canny(img, 100, 200)
    return np.count_nonzero(edges)


def canny_area_dependent_method(img):
    edges = cv2.Canny(img, 100, 200)
    return np.count_nonzero(edges) * 1000. / (edges.shape[0] * edges.shape[1])


def motion_kernel(angleRad, d, sz=65):
    angle = math.radians(angleRad)
    kern = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
    kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
    return kern/np.sum(kern)

def LineKernel(dim, angle, linetype = 'right'):
    kernelwidth = dim
    kernelCenter = int(math.floor(dim/2))
    angle = SanitizeAngleValue(kernelCenter, angle)
    kernel = np.zeros((kernelwidth, kernelwidth), dtype=np.float32)
    lineDict = LineDictionary()
    lineAnchors = lineDict.lines[dim][angle]
    if(linetype == 'right'):
        lineAnchors[0] = kernelCenter
        lineAnchors[1] = kernelCenter
    if(linetype == 'left'):
        lineAnchors[2] = kernelCenter
        lineAnchors[3] = kernelCenter
    rr,cc = draw.line(lineAnchors[0], lineAnchors[1], lineAnchors[2], lineAnchors[3])
    kernel[rr,cc]=1
    normalizationFactor = np.count_nonzero(kernel)
    kernel = kernel / normalizationFactor
    return kernel

def SanitizeAngleValue(kernelCenter, angle):
    numDistinctLines = kernelCenter * 4
    angle = math.fmod(angle, 180.0)
    validLineAngles = np.linspace(0,180, numDistinctLines, endpoint = False)
    angle = nearestValue(angle, validLineAngles)
    return angle

def nearestValue(theta, validAngles):
    idx = (np.abs(validAngles-theta)).argmin()
    return validAngles[idx]

def randomAngle(kerneldim):
    kernelCenter = int(math.floor(kerneldim/2))
    numDistinctLines = kernelCenter * 4
    validLineAngles = np.linspace(0,180, numDistinctLines, endpoint = False)
    angleIdx = np.random.randint(0, len(validLineAngles))
    return int(validLineAngles[angleIdx])


original = cv2.imread(r'lure\z13-2-68.0-5859.0-90.0-8.0.png')
# original = cv2.imread(r'lure\z13-2-1.2627435457711202-23.08679276123039.png')

psf = np.array(motion_kernel(math.degrees(1.2627435457711202), 23, sz=9))#.astype(np.float64)         # np.ones((5, 5)) / 25


nb = cv2.imread(r'lure\z19-44-302.0-23463.0-122.0-3.0.png')
# nb = cv2.imread(r'lure\z19-44-2.191045812777718-4.301162633521313.png')
gray = cv2.cvtColor(nb, cv2.COLOR_BGR2GRAY)

frame = color.rgb2gray(nb)
original_frame = color.rgb2gray(original)

blurred = cv2.filter2D(nb, -1, psf)
gray_blurred = color.rgb2gray(blurred)

psf2 = cv2.imread(r'psf.png', 0)


# cv2.imshow('img', blurred)
# cv2.waitKey()
#
# cv2.destroyAllWindows()



psf3 = np.ones((5, 5)).astype(np.float64) / 25

# deconvolved = restoration.denoise_wavelet(original, multichannel=True)
# deconvolved2 = restoration.denoise_bilateral(original, sigma_color=0.05, sigma_spatial=15, multichannel=True)
# deconvolved4 = restoration.denoise_wavelet(original, multichannel=True, convert2ycbcr=True)

# deconv = restoration.richardson_lucy(original_frame, psf, iterations=50)
# psf_est = estimate_psf(frame, blurred, psf3)
deconv = recover_image(gray_blurred, gray_blurred, psf.astype(np.float64),
                                 verbose=True, w0=50,
                                 lambda1=1.0, lambda2=20, a=1.0e+3, maxiter=5, t=5, method="gd", alpha_0=1e-6)

# deconvolved = restoration.wiener(original_frame, psf, 0.1)
# deconvolved = restoration.richardson_lucy(frame, psf, iterations=500)
deconvolved = restoration.unsupervised_wiener(original_frame, psf)

cv2.imshow('im', deconv)
cv2.waitKey()
cv2.destroyAllWindows



def SRRestore(camera, origImg, samples, upscale, iter):
    error = 0

    high_res_new = numpy.asarray(origImg).astype(numpy.float32)

    # for every LR with known pixel-offset
    for (offset, captured) in samples:

        (dx,dy) = offset

        # make LR of HR given current pixel-offset
        simulated = camera.take(origImg, offset, 1.0/upscale)

        # convert captured and simulated to numpy arrays (mind the data type!)
        cap_arr = numpy.asarray(captured).astype(numpy.float32)
        sim_arr = numpy.asarray(simulated).astype(numpy.float32)

        # get delta-image/array: captured - simulated
        delta = (cap_arr - sim_arr) / len(samples)

        # Sum of Absolute Difference Error
        error += numpy.sum(numpy.abs(delta))

        # upsample delta to HR size (with zeros)
        delta_hr_R = numpy.apply_along_axis(
                    lambda row: upsample(row,upscale),
                    1,
                    numpy.apply_along_axis(
                        lambda col: upsample(col,upscale),
                        0,
                        delta[:,:,0]))

        delta_hr_G = numpy.apply_along_axis(
                    lambda row: upsample(row,upscale),
                    1,
                    numpy.apply_along_axis(
                        lambda col: upsample(col,upscale),
                        0,
                        delta[:,:,1]))

        delta_hr_B = numpy.apply_along_axis(
                    lambda row: upsample(row,upscale),
                    1,
                    numpy.apply_along_axis(
                        lambda col: upsample(col,upscale),
                        0, delta[:,:,2]))

        # apply the offset to the delta
        delta_hr_R = camera.doOffset(delta_hr_R, (-dx,-dy))
        delta_hr_G = camera.doOffset(delta_hr_G, (-dx,-dy))
        delta_hr_B = camera.doOffset(delta_hr_B, (-dx,-dy))

        # Blur the (upsampled) delta with PSF
        delta_hr_R = camera.Convolve(delta_hr_R)
        delta_hr_G = camera.Convolve(delta_hr_G)
        delta_hr_B = camera.Convolve(delta_hr_B)

        # and update high_res image with filter result
        high_res_new += numpy.dstack((delta_hr_R,
                                      delta_hr_G,
                                      delta_hr_B))

    # normalize image array again (0-255)
    high_res_new = cliparray(high_res_new)

    return Image.fromarray(numpy.uint8(high_res_new)), error
