#!/usr/bin/python
# -*- coding:utf8 -*-

import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
import os
import threading
import warnings
import progressbar
import sys
import time
from .usr_FileIo import *

def usr_image_rotate(x, rg, row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='nearest', cval=0.):
    h, w = x.shape[row_axis], x.shape[col_axis]
    if len(x.shape) < 3:
        x = np.reshape(x,[h,w,1])
    theta = np.pi / 180 * rg
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    transform_matrix = usr_transform_matrix_offset_center(rotation_matrix,h,w)
    rot_x = usr_apply_transform(x,transform_matrix,channel_axis, fill_mode, cval)
    return rot_x

def usr_cmpl_image_rotate(x, rg, row_axis=0, col_axis=1, channel_axis=2,
                          fill_mode='nearest', cval=0.):
    real_x = np.real(x)
    imag_x = np.imag(x)
    rot_real = usr_image_rotate(real_x, rg, row_axis,col_axis,channel_axis,fill_mode,cval)
    rot_imag = usr_image_rotate(imag_x, rg, row_axis,col_axis,channel_axis,fill_mode,cval)
    rot_x = np.asarray(rot_real,dtype= np.complex64) + 1j* np.asarray(rot_imag,dtype= np.complex64)
    return rot_x


def usr_image_random_rotate(x, rg, row_axis=0, col_axis=1, channel_axis=2,
                            fill_mode='nearest', cval=0.):
    theta = np.random.uniform(-rg, rg)
    if np.iscomplexobj(x):
        return usr_cmpl_image_rotate(x,theta,row_axis,col_axis,channel_axis,fill_mode,cval)
    else:
        return usr_image_rotate(x,theta,row_axis,col_axis,channel_axis,fill_mode,cval)

def usr_image_shift(x, wrg, hrg, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    h, w = x.shape[row_axis], x.shape[col_axis]
    if len(x.shape) < 3:
        x.reshape([h, w, 1])
    tx = hrg * h #np.random.uniform(-hrg, hrg) * h
    ty = wrg * w #np.random.uniform(-wrg, wrg) * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    transform_matrix = translation_matrix  # no need to do offset
    x = usr_apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x
def usr_cmpl_image_shift(x, wrg, hrg, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    real_x = np.real(x)
    imag_x = np.imag(x)
    shift_real = usr_image_shift(real_x, wrg, hrg, row_axis, col_axis, channel_axis, fill_mode, cval)
    shift_imag = usr_image_shift(imag_x, wrg, hrg, row_axis, col_axis, channel_axis, fill_mode, cval)
    shift_x = np.asarray(shift_real, dtype=np.complex64) + 1j * np.asarray(shift_imag, dtype=np.complex64)
    return shift_x

def usr_random_image_shift(x, wrg, hrg, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    tx = np.random.uniform(-hrg, hrg)
    ty = np.random.uniform(-wrg, wrg)
    if np.iscomplexobj(x):
        return usr_cmpl_image_shift(x,tx,ty,row_axis,col_axis,channel_axis,fill_mode,cval)
    else:
        return usr_image_shift(x,tx,ty,row_axis,col_axis,channel_axis,fill_mode,cval)


def usr_image_shear(x, shear, row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='nearest', cval=0.):
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = usr_transform_matrix_offset_center(shear_matrix, h, w)
    x = usr_apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def usr_cmpl_image_shear(x, shear, row_axis=0, col_axis=1, channel_axis=2,
                         fill_mode='nearest', cval=0.):
    real_x = np.real(x)
    imag_x = np.imag(x)
    shear_real = usr_image_shear(real_x,shear,row_axis, col_axis, channel_axis,fill_mode,cval)
    shear_imag = usr_image_shear(imag_x,shear,row_axis, col_axis, channel_axis,fill_mode,cval)
    shear_x = np.asarray(shear_real, dtype=np.complex64) + 1j * np.asarray(shear_imag, dtype=np.complex64)
    return shear_x

def usr_random_image_shear(x, intensity, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    shear = np.random.uniform(-intensity, intensity)
    if np.iscomplexobj(x):
        return usr_cmpl_image_shear(x,shear,row_axis, col_axis, channel_axis,fill_mode,cval)
    else:
        return usr_image_shear(x,shear,row_axis, col_axis, channel_axis,fill_mode,cval)

def usr_apply_transform(x,transform_matrix, channel_axis = 2,
                        fill_mode='nearest', cval=0.):
    final_affine_matrix = transform_matrix[0:2, 0:2]
    final_offset = transform_matrix[0:2, 2]
    x = np.rollaxis(x, channel_axis, 0)
    channel_images = [ndi.interpolation.affine_transform(x_channel,
                                                         final_affine_matrix,
                                                         final_offset,
                                                         order=0,
                                                         mode=fill_mode,
                                                         cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=channel_axis)
    return x

def usr_transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def usr_flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def usr_random_hflip(x, threshold =0.5,img_col_axis = 1):
    if np.random.random() < threshold:
        return usr_flip_axis(x, img_col_axis)
    else:
        return x

def usr_random_vflip(x,threshold =0.5, img_row_axis = 0):
    if np.random.random() < threshold:
        return usr_flip_axis(x, img_row_axis)
    else:
        return x

def usr_random_transform(x,row_axis = 0, col_axis = 1, channel_axis = 2,
                         img_rot_angle = 0., img_shift_tx = 0., img_shift_ty = 0.,
                         intensity = 0., horizontal_flip = False, vertical_flip = False):
    aug_img = []
    if img_rot_angle != 0.:
        rot_x = usr_image_random_rotate(x,img_rot_angle,row_axis= row_axis, col_axis= col_axis,
                                        channel_axis= channel_axis,fill_mode= 'nearest',cval= 0.)
        aug_img.append(rot_x)
    if img_shift_tx != 0. or img_shift_ty != 0.:
        shift_x = usr_random_image_shift(x,img_shift_tx,img_shift_ty,row_axis= row_axis, col_axis= col_axis,
                                         channel_axis= channel_axis,fill_mode= 'nearest',cval= 0.)
        aug_img.append(shift_x)
    if intensity != 0.:
        shear_x = usr_random_image_shear(x,intensity,row_axis= row_axis, col_axis= col_axis,channel_axis= channel_axis,
                                         fill_mode= 'nearest', cval= 0.)
        aug_img.append(shear_x)
    if horizontal_flip == True:
        hflip_x = usr_flip_axis(x, col_axis)
        aug_img.append(hflip_x)
    if vertical_flip == True:
        vflip_x = usr_flip_axis(x,row_axis)
        aug_img.append(vflip_x)
    return aug_img

def usr_listsort(x):
    y = sorted(x)
    y_ = []
    indx_arr = np.zeros(len(x),dtype= np.int32)-1
    for i, val in enumerate(x):
        indx = y.index(val)
        if indx in indx_arr:
            cnt = y_.count(val)
            indx_arr[indx+cnt] = i
        else:
            indx_arr[indx] = i
        y_.append(val)
    return y,indx_arr

def find_Az_interval(x,Az_arr):
    diff_arr = np.asanyarray(Az_arr) - x
    diff_arr = np.abs(diff_arr)
    diff_list = diff_arr.tolist()
    min_val = min(diff_list)
    min_pos = diff_list.index(min_val)
    if Az_arr[min_pos] > x:
        left_pos = min_pos-1
        right_pos = min_pos
    else:
        left_pos = min_pos
        right_pos =min_pos + 1
    if left_pos < 0:
        left_pos = len(Az_arr) -1
        left_val = Az_arr[left_pos] -360
    else:
        left_val = Az_arr[left_pos]
    if right_pos >= len(Az_arr):
        right_pos = 0
        right_val = Az_arr[right_pos] +360
    else:
        right_val = Az_arr[right_pos]
    return left_pos,right_pos,left_val,right_val

def get_pose_synposis(left_az,right_az,left_img,right_img, cur_az):
    # rotate the left and right images clockwise
    if np.iscomplexobj(left_img):
        R_theta_a = usr_cmpl_image_rotate(left_img,left_az - cur_az)
        R_theta_b = usr_cmpl_image_rotate(right_img,cur_az- right_az)
    else:
        R_theta_a = usr_image_rotate(left_img,left_az - cur_az)
        R_theta_b = usr_image_rotate(right_img, cur_az- right_az)
    # synthesis new image by interpretation
    alpha_left = np.abs(cur_az - left_az)
    alpha_right = np.abs(cur_az-right_az)
    syn_img = (alpha_left* R_theta_a+ alpha_right * R_theta_b)/(alpha_left + alpha_right)
    return syn_img
def usr_random_pose_synthesis(x,Az):
    sort_Az, Az_indx = usr_listsort(Az)
    syn_angle = np.random.uniform(0,360.0,1)
    syn_angle = syn_angle[0]
    lpos, rpos, lval, rval = find_Az_interval(syn_angle, sort_Az)
    left_img =  x[Az_indx[lpos]]
    right_img = x[Az_indx[rpos]]
    syn_img = get_pose_synposis(lval,rval,left_img,right_img,syn_angle)
    return syn_img,syn_angle

def MSTAR_augment(x,y,Az,row_axis = 0, col_axis =1, channel_axis = 2,
                  nrow=128, ncol=128, nchannels = 1,
                  pose_syn = 0, rot_num =0, rot_angle = 0.,
                  shift_num =0, shift_tx =0., shift_ty =0.,
                  shear_num =0, shear_intensity = 0.,
                  horizontal_flip = False, vertical_flip = False):
    img_num = len(y)
    increment_num = 1 + rot_num + shift_num + shear_num + 2 + pose_syn  # pose_syn+ rot_num + shift_num+ shear_num + 2 + 1
    aug_img_num = img_num * (increment_num)
    aug_imgs = []
    aug_y = []
    for i in range(img_num):
        # original image format transform
        mag_img = x[i]
        aug_imgs.append(mag_img)
        aug_y.append(y[i])
        # create new image data via rotating
        if rot_num > 0:
            for j in range(rot_num):
                rot_x = usr_image_random_rotate(mag_img, rot_angle, row_axis=row_axis, col_axis=col_axis,
                                                channel_axis=channel_axis, fill_mode='nearest', cval=0.)
                aug_imgs.append(rot_x)
                aug_y.append(y[i])
        # create new image data via shifting
        if shift_num > 0:
            for j in range(shift_num):
                shift_x = usr_random_image_shift(mag_img, shift_tx, shift_ty, row_axis=row_axis, col_axis=col_axis,
                                                 channel_axis=channel_axis, fill_mode='nearest', cval=0.)
                aug_imgs.append(shift_x)
                aug_y.append(y[i])
        # create new image via shearing
        if shear_num > 0:
            for j in range(shear_num):
                shear_x = usr_random_image_shear(mag_img, shear_intensity, row_axis=row_axis, col_axis=col_axis,
                                                 channel_axis=channel_axis, fill_mode='nearest', cval=0.)
                aug_imgs.append(shear_x)
                aug_y.append(y[i])
        #
        if horizontal_flip == True:
            hflip_x = usr_flip_axis(mag_img, col_axis)
            aug_imgs.append(hflip_x)
            aug_y.append(y[i])

        if vertical_flip == True:
            vflip_x = usr_flip_axis(mag_img, row_axis)
            aug_imgs.append(vflip_x)
            aug_y.append(y[i])

        # create new image via synthesis
        if pose_syn > 0:
            for i in range(pose_syn):
                pose_syn_x = usr_random_pose_synthesis(x, Az)
                aug_imgs.append(pose_syn_x)
                aug_y.append(y[i])
        return aug_imgs, aug_y

def image_wgn(x, snr):
    img_shape = np.shape(x)
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/np.size(x)
    npower = xpower / snr
    return np.random.randn(img_shape[0],img_shape[1]) * np.sqrt(npower)

def MSTAR_Preprocesss(filepathName,pose_syn_ratio = 3,
                      rot_num =3, rot_angle = 30.,
                      noise_num = 3, snr = 20.):
    print('Reading and apply preprocessing to the MSTAR data %s'%filepathName)
    print('')
    data_path = get_MSTAR_subfolder(filepathName, file_ext='.raw')
    aug_imgs = []
    aug_TTypes = []
    aug_Sernums = []
    aug_Az = []
    for fileid, subfolder in enumerate(data_path):
        print('Loading training data samples from %s'%subfolder)
        print('')
        label_TType, label_Sernum, Az_arr, import_images = batch_read_mstar_data(subfolder)
        img_num = len(label_TType)
        print('Now apply augmentation to files in folder %s'%(subfolder))
        print('')
        bar = progressbar.ProgressBar(max_value=img_num, redirect_stdout=True)
        for i in range(img_num):
            org_img = import_images[i]
            aug_imgs.append(np.array(org_img))
            aug_TTypes.append(label_TType[i])
            aug_Sernums.append(label_Sernum[i])
            aug_Az.append(Az_arr[i])
            #rotate the orginal image to create new samples
            for rot_id in range(rot_num):
                rot_img = usr_image_random_rotate(org_img,rot_angle,row_axis= 0, col_axis= 1, channel_axis= 2,
                                                  fill_mode= 'nearest', cval= 0.0)
                rot_img = np.reshape(rot_img,newshape= [128,128])
                aug_imgs.append(np.array(rot_img))
                aug_TTypes.append(label_TType[i])
                aug_Sernums.append(label_Sernum[i])
                aug_Az.append(Az_arr[i])
            #add noise to the image to create new images
            for noise_id in range(noise_num):
                noise_factor = image_wgn(org_img,snr=snr)
                noise_img = noise_factor+org_img
                noise_img = np.reshape(noise_img,newshape= [128,128])
                aug_imgs.append(np.array(noise_img))
                aug_TTypes.append(label_TType[i])
                aug_Sernums.append(label_Sernum[i])
                aug_Az.append(Az_arr[i])
            bar.update(i)
        bar.finish()
        #create new samples by synthesize images from different
        print('Now apply synthesizing to the input image array')
        syn_num = pose_syn_ratio * img_num
        bar = progressbar.ProgressBar(max_value=syn_num, redirect_stdout=True)
        for syn_id in range(syn_num):
            pose_syn_x,syn_Az = usr_random_pose_synthesis(import_images, Az_arr)
            pose_syn_x = np.reshape(pose_syn_x,newshape= [128,128])
            aug_imgs.append(np.array(pose_syn_x))
            aug_TTypes.append(label_TType[0])
            aug_Sernums.append(label_Sernum[0])
            aug_Az.append(syn_Az)
            bar.update(syn_id)
        bar.finish()
    return aug_TTypes,aug_Sernums,aug_Az,aug_imgs




