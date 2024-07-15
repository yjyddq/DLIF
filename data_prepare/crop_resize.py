from PIL import Image
import cv2
import shutil
import os
from glob import glob
import numpy as np
import math

'''Utilize the MTCNN algorithm to detect faces, obtain bounding boxes,
    then crop the face from scene, resize to 256*256 for landmarks prediction.
    refer to the github url: https://github.com/mayuanjason/MTCNN_face_detection_alignment_pytorch | or
    https://github.com/TropComplique/mtcnn-pytorch
'''

def crop_resize_face_from_scene(image, face_name_full, scale, size):
    f = open(face_name_full, 'r')
    lines = f.readlines()
    y1, x1, y2, x2 = [float(ele) for ele in lines[:4]]
    f.close()
    w = y2 - y1
    h = x2 - x1
    y_mid = (y1 + y2) / 2.0
    x_mid = (x1 + x2) / 2.0
    h_img, w_img = image.shape[0], image.shape[1]
    w_scale = scale * w
    h_scale = scale * h
    y1 = y_mid - w_scale / 2.0
    x1 = x_mid - h_scale / 2.0
    y2 = y_mid + w_scale / 2.0
    x2 = x_mid + h_scale / 2.0
    y1 = max(math.floor(y1), 0)
    x1 = max(math.floor(x1), 0)
    y2 = min(math.floor(y2), w_img)
    x2 = min(math.floor(x2), h_img)
    region = image[x1:x2, y1:y2]
    roi = cv2.resize(region,size)
    return roi


'''crop resize and save'''
def Crop_Resize(input_root,output_root,scale=1.3,scale_up=1.6,scale_down=1.1,size=(256,256)):
    bbx_list = sorted(glob(os.path.join(input_root, "*.dat")),key = os.path.os.path.getctime)
    bbx_list_len = len(bbx_list)
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    print(bbx_list_len)
    for i in range(bbx_list_len):
        # scale = np.random.randint(int(scale_down * 10),
        #                            int(scale_up * 10))  # a num from scale down to scale up
        # scale = scale / 10.0
        img_i_path = bbx_list[i].replace('dat', 'jpg')
        print(img_i_path)
        img_i = cv2.imread(img_i_path)
        img_i_cropped = crop_resize_face_from_scene(img_i, bbx_list[i], scale, size)
        img_i_name = img_i_path.split('/')[-1][0:-4] + '_crop_resize.jpg'
        # print(img_i_name)
        output_path = os.path.join(output_root,img_i_name)

        cv2.imwrite(output_path, img_i_cropped)
    print('Successfully !')




'''C'''
# train_set
input_root = r'/mnt/g/FAS_Dataset/OCMI/CASIA_FASD/train_img_CASIA'
output_root = r'/mnt/g/FAS_Dataset/OCMI/CASIA_FASD/train_img_CASIA_crop_resize'
Crop_Resize(input_root,output_root)
# test_set
input_root = r'/mnt/g/FAS_Dataset/OCMI/CASIA_FASD/val_img_CASIA'
output_root = r'/mnt/g/FAS_Dataset/OCMI/CASIA_FASD/val_img_CASIA_crop_resize'
Crop_Resize(input_root,output_root)

'''O'''
# train_set
input_root = r'/mnt/g/FAS_Dataset/OCMI/Oulu_NPU/train_img_Oulu'
output_root = r'/mnt/g/FAS_Dataset/OCMI/Oulu_NPU/train_img_Oulu_crop_resize'
Crop_Resize(input_root,output_root)
# test_set
input_root = r'/mnt/g/FAS_Dataset/OCMI/Oulu_NPU/val_img_Oulu'
output_root = r'/mnt/g/FAS_Dataset/OCMI/Oulu_NPU/val_img_Oulu_crop_resize'
Crop_Resize(input_root,output_root)

'''I'''
# train_set
input_root = r'/mnt/g/FAS_Dataset/OCMI/Replay_Attack/train_img_ReplayAttack'
output_root = r'/mnt/g/FAS_Dataset/OCMI/Replay_Attack/train_img_ReplayAttack_crop_resize'
Crop_Resize(input_root,output_root)
# test_set
input_root = r'/mnt/g/FAS_Dataset/OCMI/Replay_Attack/val_img_ReplayAttack'
output_root = r'/mnt/g/FAS_Dataset/OCMI/Replay_Attack/val_img_ReplayAttack_crop_resize'
Crop_Resize(input_root,output_root)

'''M'''
# train_set
input_root = r'/mnt/g/FAS_Dataset/OCMI/MSU_MFSD/train_img_MSU'
output_root = r'/mnt/g/FAS_Dataset/OCMI/MSU_MFSD/train_img_MSU_crop_resize'
Crop_Resize(input_root,output_root)
# test_set
input_root = r'/mnt/g/FAS_Dataset/OCMI/MSU_MFSD/val_img_MSU'
output_root = r'/mnt/g/FAS_Dataset/OCMI/MSU_MFSD/val_img_MSU_crop_resize'
Crop_Resize(input_root,output_root)





