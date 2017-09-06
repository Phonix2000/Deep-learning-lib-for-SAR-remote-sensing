#!/usr/bin/python
# -*- coding:utf8 -*-

import numpy as np
import os as os
import tensorflow as tf
import struct
import matplotlib.pyplot as plt
import progressbar
import sys
import time

'''***************************************************************************
                        Direction operators
****************************************************************************'''
def get_all_files_in_rootfolder(root_folder, file_ext = '.raw'):
    data_path = []
    for dirpath, dirs, files in os.walk(root_folder):
        for name in files:  # files保存的是所有的文件名
            if os.path.splitext(name)[1] == file_ext:
                filename = os.path.join(dirpath, name)  # 加上路径，dirpath是遍历时文件对应的路径
                data_path.append(filename)
    return data_path

def get_MSTAR_subfolder(root_folder, file_ext = '.raw'):
    subfolders = []
    prefolder = []
    for dirpath, dirs, files in os.walk(root_folder):
        for name in files:  # files保存的是所有的文件名
            if os.path.splitext(name)[1] == file_ext:
                if prefolder != dirpath:
                    subfolders.append(dirpath)
                    prefolder = dirpath
    return subfolders

'''***************************************************************************
                        MSTAR data Reading and writing
****************************************************************************'''
def read_mstar_data(full_path_name,nrow =128, ncol =128,datamode = 'complex'):
    '''Read single data file from file'''
    f = open(full_path_name,'rb')
    amp = np.ones([nrow*ncol],dtype=np.float32)
    phase = np.ones([nrow * ncol], dtype=np.float32)
    count = 0
    while True:
        tmp_byte = f.read(4)
        count += 1
        if  count < nrow*ncol:
            pixel_amp, = struct.unpack('>f', tmp_byte)
            amp[count] = pixel_amp
        elif count < 2*nrow*ncol:
            pixel_phase, = struct.unpack('>f', tmp_byte)
            phase[count - nrow*ncol] = pixel_phase
        else:
            break
    f.close()
    amp = amp.reshape([nrow,ncol])
    phase = phase.reshape([nrow,ncol])
    '''reform complex data'''
    complex_img = np.empty([nrow,ncol], dtype=np.complex64)
    for i in range(0,nrow,1):
        for j in range(0,ncol,1):
            real_part = amp[i,j] * np.cos(phase[i,j])
            imag_part = amp[i,j] * np.sin(phase[i,j])
            complex_img[i,j] = complex(real= real_part,imag=imag_part)
    if datamode == 'complex':
        return complex_img
    elif datamode == 'multi_channel':
        multchan_img = np.zeros([nrow,ncol,2])
        multchan_img[:,:,0] = np.real(complex_img)
        multchan_img[:,:,1] = np.imag(complex_img)
        return multchan_img
    elif datamode == 'amplitude_phase':
        multchan_img = np.zeros([nrow, ncol, 2])
        multchan_img[:, :, 0] = amp
        multchan_img[:, :, 1] = phase
        return multchan_img
    else:
        return amp

def read_mstar_header(data_file_path):
    header_path = data_file_path[:-12] + '.hdr'
    f = open(header_path, 'r')
    SerNum = ''
    TType = ''
    Azangle = 0.
    while True:
        infostr = f.readline()
        if len(infostr) <= 0:
            break
        else:
            if infostr.split('=')[0] == 'TargetType':
                TType = infostr.split()[1]
            if infostr.split('=')[0] == 'TargetSerNum':
                SerNum = infostr.split()[1]
            if infostr.split('=')[0] == 'TargetAz':
                Azangle = float(infostr.split()[1])
    f.close()
    return TType,SerNum,Azangle

def batch_read_mstar_data(source_root_path, nrow=128, ncol=128, nchannels=2,import_mode = 'complex'):
    data_path = get_all_files_in_rootfolder(source_root_path, file_ext='.raw')
    x = []
    label_TType = []
    label_Sernum = []
    Az_arr = []
    import_images = []
    bar = progressbar.ProgressBar(max_value= len(data_path), redirect_stdout=True)
    for i, file_path_name in enumerate(data_path):
        #print('Now reading %s, no. %d  in %d files' % (file_path_name, i + 1, len(data_path)))
        import_image = read_mstar_data(file_path_name,nrow,ncol,import_mode)
        import_images.append(import_image)
        TType, SerNum,Az = read_mstar_header(file_path_name)
        label_TType.append(TType)
        label_Sernum.append(SerNum)
        Az_arr.append(Az)
        bar.update(i)
    bar.finish()
    return label_TType, label_Sernum,Az_arr, import_images


def convert_MSTAR_to_realinputLayer(x,y,nrow =128, ncol =128, nchannels =1, str = ''):
    print(str)
    tnum = len(y)
    bar = progressbar.ProgressBar(max_value=tnum, redirect_stdout=True)
    new_x = np.zeros((tnum, nrow, ncol, nchannels))
    new_y = usr_atoi_three_type_MSTAR(y)
    for i in range(tnum):
        mag_img = np.abs(x[i])
        new_x[i, :, :, :] = np.abs(np.reshape(mag_img, newshape=(nrow, ncol, nchannels)))
        bar.update(i)
    bar.finish()
    return new_x,new_y
'''***************************************************************************
                        Convert data type
****************************************************************************'''
def int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def bytes_feature(value):
    return  tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value= [value]))


def usr_atoi_three_type_MSTAR(train_ty):
    train_tlabels = []
    for i, ttype in enumerate(train_ty):
        if ttype == 'bmp2_tank':
            train_ty_label = 1
        elif ttype == 'btr70_transport':
            train_ty_label = 2
        elif ttype == 't72_tank':
            train_ty_label = 3
        else:
            print('Undefined target type')
            train_ty_label = 0
        train_tlabels.append(train_ty_label)
    return train_tlabels

def usr_itoa_three_type_MSTAR(train_ty):
    train_talabels = []
    for i, ttype in enumerate(train_ty):
        if ttype == 1:
            train_ty_label = 'bmp2_tank'
        elif ttype == 2:
            train_ty_label = 'btr70_transport'
        elif ttype == 3:
            train_ty_label = 't72_tank'
        else:
            print('Undefined target type')
            train_ty_label = 'Undefined'
        train_talabels.append(train_ty_label)
    return train_talabels

def usr_atoi_type_sernum_MSTAR(target_type,serial_num):
    sernum_labels = []
    ttype_labels = []
    for ttype, sernum in zip(target_type,serial_num):
        if ttype == 'bmp2_tank':
            tt_label = 1
            if sernum == '9563':
                sn_label =10
            elif sernum == '9566':
                sn_label = 11
            elif sernum == 'c21':
                sn_label = 12
            else:
                sn_label = 19
        elif ttype == 'btr70_transport':
            tt_label = 2
            if sernum == 'c71':
                sn_label = 20
            else:
                sn_label =29
        elif ttype == 't72_tank':
            tt_label = 3
            if sernum == '132':
                sn_label = 30
            elif sernum == '812':
                sn_label = 31
            elif sernum == 's7':
                sn_label = 32
            else:
                sn_label = 39
        else:
            tt_label = 0
            sn_label = 0
        ttype_labels.append(tt_label)
        sernum_labels.append(sn_label)
    return ttype_labels,sernum_labels

def usr_itoa_type_sernum_MSTAR(ttype_labels,sernum_labels):
    target_type = []
    serial_num = []
    for tt_label, sn_label in zip(ttype_labels,sernum_labels):
        if tt_label == 1:
            type_label = 'bmp2_tank'
            if sn_label == 10:
                sernum = '9563'
            elif sn_label == 11:
                sernum = '9566'
            elif sn_label == 12:
                sernum = 'c21'
            else:
                sernum = 'unknown'
        elif tt_label == 2:
            type_label = 'btr70_transport'
            if sn_label == 20:
                sernum = 'c71'
            else:
                sernum = 'unknown'
        elif tt_label == 3:
            type_label = 't72_tank'
            if sn_label == 30:
                sernum = '132'
            elif sn_label == 31:
                sernum = '812'
            elif sn_label == 32:
                sernum = 's7'
            else:
                sernum = 'unknown'
        else:
            type_label = 'unknown'
            sernum = 'unknown'
        target_type.append(type_label)
        serial_num.append(sernum)
    return target_type,serial_num

def Export_MSTAR_to_tfrecord(x,Ttype,Tsernum,TAz,filepath):
    print('export the complex dataset into tfrecord file:%s' % filepath)
    num_data = len(Ttype)
    TType_arr, Sernum_arr = usr_atoi_type_sernum_MSTAR(Ttype,Tsernum)
    writer = tf.python_io.TFRecordWriter(filepath)
    procbar = progressbar.ProgressBar(max_value= num_data, redirect_stdout=True)
    for i in range(num_data):
        img_x = x[i]
        real_x = np.real(img_x)
        imag_x = np.imag(img_x)
        type_y = TType_arr[i]
        sernum_y = Sernum_arr[i]
        az_y = TAz[i]
        example = tf.train.Example(features = tf.train.Features(
                                feature={'Type_label':int64_feature(type_y),
                                         'Sernum_label':int64_feature(sernum_y),
                                         'Azimuth_angle': bytes_feature(az_y.tostring()),
                                         'real_image_raw':bytes_feature(real_x.tostring()),
                                         'imag_image_raw':bytes_feature(imag_x.tostring())})
        )
        writer.write(example.SerializeToString())
        procbar.update(i)
    writer.close()
    procbar.finish()

def Export_MSTAR_to_NumpyFile(x,Ttype,Tsernum,TAz,filepath):
    if filepath.split('.')[-1] != 'npy':
        Datafilepath = filepath.split('.')[-2]+'.npy'
        Headfilepath = filepath.split('.')[-2]+'_head.npy'
    else:
        Datafilepath = filepath
        Headfilepath = filepath.split('.')[-2] + '_head.npy'
    print('export the complex dataset into tfrecord file:%s' % filepath)
    TType_arr, Sernum_arr = usr_atoi_type_sernum_MSTAR(Ttype, Tsernum)
    TType_arr = np.array(TType_arr)
    Sernum_arr = np.array(Sernum_arr)
    TAz = np.array(TAz)
    x = np.array(x)
    np.save(Headfilepath,[TType_arr,Sernum_arr,TAz])
    np.save(Datafilepath,x)

def Load_MSTAR_from_NumpyFile(filepath):
    if filepath.split('.')[-1] != 'npy':
        print('Invalid numpy data file'%filepath)
        return -1
    else:
        Datafilepath = filepath
        Headfilepath = filepath.split('.')[-2] +'_head.npy'
    TType_arr, Sernum_arr, TAz = np.load(Headfilepath)
    x = np.load(Datafilepath)
    TType_arr = TType_arr.tolist()
    Sernum_arr = Sernum_arr.tolist()
    TAz = TAz.tolist()
    x = x.tolist()
    return TType_arr,Sernum_arr,TAz,x
















