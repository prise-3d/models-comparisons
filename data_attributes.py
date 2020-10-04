# main imports
import numpy as np
import sys

# image transform imports
from PIL import Image
from skimage import color, restoration
from sklearn.decomposition import FastICA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import TruncatedSVD
from numpy.linalg import svd as lin_svd
from scipy.signal import medfilt2d, wiener, cwt
import pywt
import cv2

from ipfml.processing import transform, compression, segmentation
from ipfml import utils

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from modules.utils import data as dt


def get_image_features(data_type, block):
    """
    Method which returns the data type expected
    """
    
    if 'filters_statistics' in data_type:

        img_width, img_height = 200, 200

        lab_img = transform.get_LAB_L(block)
        arr = np.array(lab_img)

        # compute all filters statistics
        def get_stats(arr, I_filter):

            e1       = np.abs(arr - I_filter)
            L        = np.array(e1)
            mu0      = np.mean(L)
            A        = L - mu0
            H        = A * A
            E        = np.sum(H) / (img_width * img_height)
            P        = np.sqrt(E)

            return mu0, P
            # return np.mean(I_filter), np.std(I_filter)

        stats = []

        kernel = np.ones((3,3),np.float32)/9
        stats.append(get_stats(arr, cv2.filter2D(arr,-1,kernel)))

        kernel = np.ones((5,5),np.float32)/25
        stats.append(get_stats(arr, cv2.filter2D(arr,-1,kernel)))

        stats.append(get_stats(arr, cv2.GaussianBlur(arr, (3, 3), 0.5)))

        stats.append(get_stats(arr, cv2.GaussianBlur(arr, (3, 3), 1)))

        stats.append(get_stats(arr, cv2.GaussianBlur(arr, (3, 3), 1.5)))

        stats.append(get_stats(arr, cv2.GaussianBlur(arr, (5, 5), 0.5)))

        stats.append(get_stats(arr, cv2.GaussianBlur(arr, (5, 5), 1)))

        stats.append(get_stats(arr, cv2.GaussianBlur(arr, (5, 5), 1.5)))

        stats.append(get_stats(arr, medfilt2d(arr, [3, 3])))

        stats.append(get_stats(arr, medfilt2d(arr, [5, 5])))

        stats.append(get_stats(arr, wiener(arr, [3, 3])))

        stats.append(get_stats(arr, wiener(arr, [5, 5])))

        wave = w2d(arr, 'db1')
        stats.append(get_stats(arr, np.array(wave, 'float64')))

        data = []

        for stat in stats:
            data.append(stat[0])

        for stat in stats:
            data.append(stat[1])
        
        data = np.array(data)

    if 'Constantin2016' in data_type:

        img_width, img_height = 200, 200

        lab_img = transform.get_LAB_L(block)
        arr = np.array(lab_img)
    
        stats = []

        kernel = np.ones((3,3),np.float32)/9
        arr = cv2.filter2D(arr,-1,kernel)

        kernel = np.ones((5,5),np.float32)/25
        arr = cv2.filter2D(arr,-1,kernel)

        arr = cv2.GaussianBlur(arr, (3, 3), 0.5)

        arr = cv2.GaussianBlur(arr, (3, 3), 1)

        arr = cv2.GaussianBlur(arr, (3, 3), 1.5)

        arr = cv2.GaussianBlur(arr, (5, 5), 0.5)

        arr = cv2.GaussianBlur(arr, (5, 5), 1)

        arr = cv2.GaussianBlur(arr, (5, 5), 1.5)

        arr = medfilt2d(arr, [3, 3])

        arr = medfilt2d(arr, [5, 5])

        arr = wiener(arr, [3, 3])

        arr = wiener(arr, [5, 5])

        wave = w2d(arr, 'db1')
        output = np.array(wave, 'float32')

        # compute abs difference between the two images
        data = np.abs(np.array(lab_img) - output)
        
        data = np.array(data.flatten())

        # if normalization by L channel is required
        if 'norm' in data_type:
            data /= 100.

    if 'lab' in data_type:

        data = transform.get_LAB_L_SVD_s(block)

    return data


def w2d(arr, mode):
    # convert to float   
    imArray = arr
    # np.divide(imArray, 100) # because of lightness channel, use of 100

    # compute coefficients 
    # same to: LL (LH, HL, HH)
    # cA, (cH, cV, cD) = pywt.dwt2(imArray, mode)
    # cA *= 0 # remove low-low sub-bands data

    # reduce noise from the others cofficients
    # LH, HL and HH
    # ----
    # cannot use specific method to predict thresholds...
    # use of np.percentile(XX, 5) => remove data under 5 first percentile
    # cH = pywt.threshold(cH, np.percentile(cH, 5), mode='soft')
    # cV = pywt.threshold(cV, np.percentile(cV, 5), mode='soft')
    # cD = pywt.threshold(cD, np.percentile(cD, 5), mode='soft')

    # reconstruction
    # imArray_H = pywt.idwt2((cA, (cH, cV, cD)), mode)
    # print(np.min(imArray_H), np.max(imArray_H), np.mean(imArray_H))
    # imArray_H *= 100 # because of lightness channel, use of 100
    # imArray_H = np.array(imArray_H)

    # coeffs = pywt.wavedec2(imArray, mode, level=2)

    # #Process Coefficients
    # coeffs_H=list(coeffs)  
    # coeffs_H[0] *= 0;  

    # # reconstruction
    # imArray_H=pywt.waverec2(coeffs_H, mode)
    # print(np.min(imArray_H), np.max(imArray_H), np.mean(imArray_H))

    # using skimage
    sigma = restoration.estimate_sigma(imArray, average_sigmas=True, multichannel=False)
    imArray_H = restoration.denoise_wavelet(imArray, sigma=sigma, wavelet='db1', mode='soft', 
        wavelet_levels=2, 
        multichannel=False, 
        convert2ycbcr=False, 
        method='VisuShrink', 
        rescale_sigma=True)

    # imArray_H *= 100

    return imArray_H


def _get_mscn_variance(block, sub_block_size=(50, 50)):

    blocks = segmentation.divide_in_blocks(block, sub_block_size)

    data = []

    for block in blocks:
        mscn_coefficients = transform.get_mscn_coefficients(block)
        flat_coeff = mscn_coefficients.flatten()
        data.append(np.var(flat_coeff))

    return np.sort(data)

