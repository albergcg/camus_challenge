import os
import numpy as np
import SimpleITK as sitk
import h5py
import cv2

train_path = "../data/training/"

def mhd_to_array(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path, sitk.sitkFloat32))

train_2ch_frames_list = sorted(os.listdir(train_path + "2ch/frames/"))
train_2ch_masks_list = sorted(os.listdir(train_path + "2ch/masks/"))
train_4ch_frames_list = sorted(os.listdir(train_path + "4ch/frames/"))
train_4ch_masks_list = sorted(os.listdir(train_path + "4ch/masks/"))

f = h5py.File("../data/image_dataset.hdf5", "w")

f.create_dataset("train 2ch frames", (900, 384, 384, 1),
                chunks = (4, 384, 384, 1), dtype = "float32")

f.create_dataset("train 2ch masks", (900, 384, 384, 1),
                chunks = (4, 384, 384, 1), dtype = "int32")

f.create_dataset("train 4ch frames", (900, 384, 384, 1),
                chunks = (4, 384, 384, 1), dtype = "float32")

f.create_dataset("train 4ch masks", (900, 384, 384, 1),
                chunks = (4, 384, 384, 1), dtype = "int32")

j = 0
for i in train_2ch_frames_list:
    if "mhd" in i:
        array = mhd_to_array(os.path.join(train_path, "2ch/frames", i))
        new_array = cv2.resize(array[0,:,:], dsize=(384, 384), interpolation=cv2.INTER_CUBIC)
        new_array = np.reshape(new_array,(384,384,1))
        new_array = new_array/255
        f["train 2ch frames"][j,...] = new_array[...]
        j = j + 1        
        
j = 0
for i in train_2ch_masks_list:
    if "mhd" in i:
        array = mhd_to_array(os.path.join(train_path, "2ch/masks", i))
        new_array = cv2.resize(array[0,:,:], dsize=(384, 384), interpolation=cv2.INTER_NEAREST)
        new_array = np.reshape(new_array,(384,384,1))
        f["train 2ch masks"][j,...] = new_array[...]
        j = j + 1   
        
j = 0
for i in train_4ch_frames_list:
    if "mhd" in i:
        array = mhd_to_array(os.path.join(train_path, "4ch/frames", i))
        new_array = cv2.resize(array[0,:,:], dsize=(384, 384), interpolation=cv2.INTER_CUBIC)
        new_array = np.reshape(new_array,(384,384,1))
        new_array = new_array/255
        f["train 4ch frames"][j,...] = new_array[...]
        j = j + 1        
        
j = 0
for i in train_4ch_masks_list:
    if "mhd" in i:
        array = mhd_to_array(os.path.join(train_path, "4ch/masks", i))
        new_array = cv2.resize(array[0,:,:], dsize=(384, 384), interpolation=cv2.INTER_NEAREST)
        new_array = np.reshape(new_array,(384,384,1))
        f["train 4ch masks"][j,...] = new_array[...]
        j = j + 1       
        
f.close()

