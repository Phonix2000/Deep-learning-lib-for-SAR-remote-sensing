#!/usr/bin/python
# -*- coding:utf8 -*-

import numpy as np
import progressbar
from scipy import linalg
from .usr_utils import *


def usr_rand_select_patches(imgs_list,npatchnum,npatchrow,npatchcol):
    img_shape = np.shape(imgs_list)
    if np.size(img_shape) == 3:
        nimg = img_shape[0]
        nrow = img_shape[1]
        ncol = img_shape[2]
        nchannel = 1
    elif np.size(img_shape) == 4:
        nimg = img_shape[0]
        nrow = img_shape[1]
        ncol = img_shape[2]
        nchannel = img_shape[3]
    else:
        print('Error(usr_rand_select_patches): Expect tuple with 3 or 4 dims, the image size is incorrect')
        return None,None
    patches_list = []
    img_indx_arr = np.random.randint(low= 0, high= nimg,size= npatchnum)
    pbar = progressbar.ProgressBar(max_value= npatchnum, redirect_stdout=True)
    for patchid in range(npatchnum):
        img_index = img_indx_arr[patchid]
        rowstart = np.random.randint(0, nrow - npatchrow + 1)
        colstart = np.random.randint(0, ncol - npatchcol + 1)
        if nchannel == 1:
            patch_image = imgs_list[img_index][rowstart:rowstart + npatchrow, colstart:colstart + npatchcol]
        else:
            patch_image = imgs_list[img_index][rowstart:rowstart + npatchrow,colstart:colstart + npatchcol,:]
        patches_list.append(patch_image)
        pbar.update(patchid)
    pbar.finish(end= '\n All patches have been extracted!')
    return patches_list,img_indx_arr

def usr_get_Subimg(imgs_list,nsubrow = 64,nsubcol =64):
    img_shape = np.shape(imgs_list)
    if np.size(img_shape) == 3:
        nimg = img_shape[0]
        nrow = img_shape[1]
        ncol = img_shape[2]
        nchannel = 1
    elif np.size(img_shape) == 4:
        nimg = img_shape[0]
        nrow = img_shape[1]
        ncol = img_shape[2]
        nchannel = img_shape[3]
    else:
        print('Error(usr_get_Subimg): Expect tuple with 3 or 4 dims, the image size is incorrect')
        return None, None
    rowmid = int(nrow/2)
    colmid = int(ncol/2)
    rowstart = max(0,rowmid - int(nsubrow/2))
    rowend = min(nrow,rowstart + nsubrow)
    colstart = max(0, colmid - int(nsubcol/2))
    colend = min(ncol,colstart + nsubcol)
    nsubrow = rowend - rowstart
    nsubcol = colend - colstart
    subimg_list = []
    pbar = progressbar.ProgressBar(max_value=nimg, redirect_stdout=True)
    for i in range(nimg):
        if nchannel == 1:
            subimg = imgs_list[i][rowstart:rowend,colstart:colend]
        else:
            subimg = imgs_list[i][rowstart:rowend, colstart:colend,:]
        subimg_list.append(subimg)
        pbar.update(i)
    pbar.finish()
    return subimg_list,nsubrow, nsubcol

def convert_patches_to_vector(patches_list):
    patches_nparr = np.array(patches_list)
    print('Convert extracted patches to vector samples for CMDAE training')
    train_vector = np.reshape(patches_nparr, [len(patches_nparr), np.prod(patches_nparr.shape[1:])])
    return train_vector

def image_contrast_normalize(patches_list, eps_val = 1e-7, minval = 0.1, maxval = 0.9):
    patches_nparr = np.array(patches_list)
    patch_shape = np.shape(patches_nparr)
    normalized_x = patches_list.copy()
    if np.size(patch_shape) < 2:
        print('Invalid patch size,please check')
        return None
    else:
        nimg = patch_shape[0]
        for img_id in range(nimg):
            curimg = normalized_x[img_id]
            mean_val = np.mean(curimg)
            curimg -= mean_val
            std_val = np.std(curimg) + eps_val
            pstd = 3* std_val
            curimg = usr_up_low_bounded(curimg,-pstd,pstd)
            curimg /= pstd
            curimg = (curimg+1)*((maxval-minval)/2) + minval
            normalized_x[img_id] = curimg
    return normalized_x


def usr_image_rescale(img_arr,minval = 0.1, maxval = 0.9):
    imgs = np.array(img_arr)
    if imgs.dtype == np.complex64 or imgs.dtype == np.complex:
        amp_arr = np.abs(imgs)
    else:
        amp_arr = imgs.copy()
    amp_arr += 1e-9
    phase_arr = imgs/amp_arr
    amp_arr = image_contrast_normalize(amp_arr,eps_val=1e-7, minval= minval, maxval= maxval)
    rescaled_imgs = amp_arr*phase_arr
    return rescaled_imgs

def usr_vectors_zca(train_vectors):
    train_vectors = np.array(train_vectors)
    flat_x = np.copy(train_vectors)
    sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
    u, s, _ = linalg.svd(sigma)
    s += 10e-7
    inv_s = 1.0 / np.sqrt(s)
    diag_mat = np.diag(inv_s, k=0)
    nimgs = train_vectors.shape[0]
    principal_components = np.dot(np.dot(u, diag_mat), u.T)
    # apply whitening to the training set and test set
    for i in range(nimgs):
        flat_x[i] = np.dot(flat_x[i], principal_components)
    return flat_x, principal_components




