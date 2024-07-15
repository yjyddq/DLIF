import cv2
import numpy as np
import pandas as pd
import shutil
import os
from glob import glob
import re
from PIL import Image

'''Split C'''
def video2img_C(in_root, outpath, interval, frames_per_video, train):
    '''
    :param input_root: video path which need to be split
    :param outpath:  splited img path
    :param interval: sample interval
    :return: None
    '''
    if train: # train set has 20 subjects
        subjects = 20
        out_dir = os.path.join(outpath, 'train_img_CASIA_FASD')
    else: # test set has 30 subjects
        subjects = 30
        out_dir = os.path.join(outpath, 'val_img_CASIA_FASD')

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    else:
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)

    rvideo_counter = 0
    avideo_counter = 0
    for i in range(1,subjects+1):
        in_dir = os.path.join(in_root,'{0}'.format(i))
        video_dir_list = sorted(glob(os.path.join(in_dir, "*.avi")),key = os.path.getctime) # return a list
        nvideo_avi = len(video_dir_list)

        for j in range(nvideo_avi):
            rframe_counter = 0
            aframe_counter = 0
            if video_dir_list[j].split('\\')[-1][:-4] in ['1','2','HR_1']: 
                rvideo_counter += 1
                cap = cv2.VideoCapture(video_dir_list[j])
                if cap.isOpened():
                    while True:
                        ret = cap.grab()
                        if not ret:
                            break
                        rframe_counter += 1
                        counter = rframe_counter
                        if counter % (interval+1) == 0:
                            if counter // (interval+1) > frames_per_video:
                                break
                            ret, frame = cap.retrieve()
                            if frame is None:
                                break
                            # domain_ID,real | attack,video_ID,frame_ID
                            imgname = "{0}_{1}_{2}_{3}.jpg".format('C','real',rvideo_counter,counter//(interval+1))
                            path = os.path.join(out_dir, imgname)
                            cv2.imwrite(path, frame)
                cap.release()
            else:
                avideo_counter += 1
                cap = cv2.VideoCapture(video_dir_list[j])
                if cap.isOpened():
                    while True:
                        ret = cap.grab()
                        if not ret:
                            break
                        aframe_counter += 1
                        counter = aframe_counter
                        if counter % (interval + 1) == 0:
                            if counter // (interval+1) > frames_per_video:
                                break
                            ret, frame = cap.retrieve()
                            if frame is None:
                                break
                            imgname = "{0}_{1}_{2}_{3}.jpg".format('C','attack',avideo_counter,counter//(interval+1))
                            path = os.path.join(out_dir, imgname)
                            cv2.imwrite(path, frame)
                cap.release()
    print('Successfully !')

'''Split O'''
def video2img_O(input_path, outpath,  interval, frames_per_video,train=True):
    '''
    :param input_path: video path which need to be split
    :param outpath:  splited img path
    :param interval: sample interval
    :return: None
    '''
    if train:
        out_dir = os.path.join(outpath, 'train_img_Oulu')
    else:
        out_dir = os.path.join(outpath, 'val_img_Oulu')

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    else:
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)

    if train: # train_files
        rvideo_counter = 0
        avideo_counter = 0
        for i in range(1, 7): 
            for j in range(1, 4):
                for n in range(1, 21): # subject_ID
                    for k in range(1, 6): # i_subject 
                        rframe_counter = 0
                        aframe_counter = 0
                        if n < 10:
                            vpath = os.path.join(input_path, '{0}_{1}_0{2}_{3}.avi'.format(i, j, n, k))
                        else:
                            vpath = os.path.join(input_path, '{0}_{1}_{2}_{3}.avi'.format(i, j, n, k))
                        cap = cv2.VideoCapture(vpath)
                        if k == 1:
                            rvideo_counter += 1
                        else:
                            avideo_counter += 1
                        if cap.isOpened():
                            while True:
                                ret = cap.grab()
                                if not ret:
                                    break
                                if k == 1:
                                    rframe_counter += 1
                                    if rframe_counter % (interval + 1) == 0:
                                        if rframe_counter // (interval+1) > frames_per_video:
                                            break
                                        ret, frame = cap.retrieve()
                                        if frame is None:
                                            break
                                        # domain_ID,real | attack,video_ID,frame_ID
                                        imgname = "{0}_{1}_{2}_{3}.jpg".format('O','real',rvideo_counter,
                                            rframe_counter // (interval + 1))
                                        path = os.path.join(out_dir, imgname)
                                        cv2.imwrite(path, frame)
                                else:
                                    aframe_counter += 1
                                    if aframe_counter % (interval + 1) == 0:
                                        if aframe_counter // (interval+1) > frames_per_video:
                                            break
                                        ret, frame = cap.retrieve()
                                        if frame is None:
                                            break
                                        # domain_ID,real | attack,video_ID,frame_ID
                                        imgname = "{0}_{1}_{2}_{3}.jpg".format('O','attack',avideo_counter,
                                            aframe_counter // (interval + 1))
                                        path = os.path.join(out_dir, imgname)
                                        cv2.imwrite(path, frame)
                        cap.release()

    else: # val_files
        rvideo_counter = 0
        avideo_counter = 0
        for i in range(1, 7):
            for j in range(1, 4):
                for n in range(36, 56):
                    for k in range(1, 6):
                        rframe_counter = 0
                        aframe_counter = 0
                        if n < 10:
                            vpath = os.path.join(input_path, '{0}_{1}_0{2}_{3}.avi'.format(i, j, n, k))
                        else:
                            vpath = os.path.join(input_path, '{0}_{1}_{2}_{3}.avi'.format(i, j, n, k))
                        if k == 1:
                            rvideo_counter += 1
                        else:
                            avideo_counter += 1
                        cap = cv2.VideoCapture(vpath)
                        if cap.isOpened():
                            while True:
                                ret = cap.grab()
                                if not ret:
                                    break
                                if k == 1:
                                    rframe_counter += 1
                                    if rframe_counter % (interval + 1) == 0:
                                        if rframe_counter // (interval+1) > frames_per_video:
                                            break
                                        ret, frame = cap.retrieve()
                                        if frame is None:
                                            break
                                        imgname = "{0}_{1}_{2}_{3}.jpg".format('O','real',rvideo_counter,
                                            rframe_counter // (interval + 1))
                                        path = os.path.join(out_dir, imgname)
                                        cv2.imwrite(path, frame)
                                else:
                                    aframe_counter += 1
                                    if aframe_counter % (interval + 1) == 0:
                                        if aframe_counter // (interval+1) > frames_per_video:
                                            break
                                        ret, frame = cap.retrieve()
                                        if frame is None:
                                            break
                                        imgname = "{0}_{1}_{2}_{3}.jpg".format('O','attack',avideo_counter,
                                                  aframe_counter // (interval + 1))
                                        path = os.path.join(out_dir, imgname)
                                        cv2.imwrite(path, frame)
                        cap.release()
    print('Successfully !')

'''Split I'''
def video2img_I(input_path, outpath, interval, frames_per_video, train):
    '''
    :param input_path: video path which need to be split
    :param outpath:  splited img path
    :param interval: sample interval
    :return: None
    '''
    rpath = os.path.join(input_path, 'real')
    apath_hand = os.path.join(input_path, 'attack', 'hand')
    apath_fixed = os.path.join(input_path, 'attack', 'fixed')
    if train:
        out_img_path = os.path.join(outpath, 'train_img_replayattack')
    else:
        out_img_path = os.path.join(outpath, 'val_img_replayattack')


    rvideo_mov = sorted(glob(os.path.join(rpath, "*.mov")),key = os.path.getctime) # return a list of path
    rnvideo_mov = len(rvideo_mov)

    avideo_mov_h = sorted(glob(os.path.join(apath_hand, "*.mov")),key = os.path.getctime)  # return a list of path
    anvideo_mov_h = len(avideo_mov_h)

    avideo_mov_f = sorted(glob(os.path.join(apath_fixed, "*.mov")),key = os.path.getctime)  # return a list of path
    anvideo_mov_f = len(avideo_mov_f)

    if not os.path.isdir(out_img_path):
        os.makedirs(out_img_path)
    else:
        shutil.rmtree(out_img_path)
        os.makedirs(out_img_path)

    pattern = r"(\D+)(\d+)"
    obj = re.compile(pattern)

    for i in range(rnvideo_mov):
        rframe_counter = 0
        cap = cv2.VideoCapture(rvideo_mov[i])
        if cap.isOpened():
            while True:
                ret = cap.grab()
                if not ret:
                    break
                rframe_counter += 1
                if rframe_counter % (interval+1) == 0:
                    if frames_per_video is not None:
                        if rframe_counter // (interval+1) > frames_per_video:
                            break
                    ret, frame = cap.retrieve()
                    if frame is None:
                        break
                    if obj.findall(rvideo_mov[i])[0][1].startswith('0'):
                        if obj.findall(rvideo_mov[i])[0][1][1:].startswith('0'):
                            id = obj.findall(rvideo_mov[i])[0][1][2:]
                        else:
                            id = obj.findall(rvideo_mov[i])[0][1][1:]
                    else:
                        id = obj.findall(rvideo_mov[i])[0][1]
                    # domain_ID,subject_ID,real | attack,video_ID,frame_ID
                    imgname = "{0}_{1}_{2}_{3}_{4}.jpg".format('I',id,'real',i,rframe_counter//(interval+1))
                    path = os.path.join(out_img_path, imgname)
                    cv2.imwrite(path, frame)
        cap.release()

    for i in range(anvideo_mov_h):
        aframe_counter = 0
        cap = cv2.VideoCapture(avideo_mov_h[i])
        if cap.isOpened():
            while True:
                ret = cap.grab()
                if not ret:
                    break
                aframe_counter += 1
                if aframe_counter % (interval+1) == 0:
                    if frames_per_video is not None:
                        if aframe_counter // (interval+1) > frames_per_video:
                            break
                    ret, frame = cap.retrieve()
                    if frame is None:
                        break
                    if obj.findall(avideo_mov_h[i])[0][1].startswith('0'):
                        if obj.findall(avideo_mov_h[i])[0][1][1:].startswith('0'):
                            id = obj.findall(avideo_mov_h[i])[0][1][2:]
                        else:
                            id = obj.findall(avideo_mov_h[i])[0][1][1:]
                    else:
                        id = obj.findall(avideo_mov_h[i])[0][1]
                    imgname = "{0}_{1}_{2}_{3}_{4}.jpg".format('I',id,'hattack',i,aframe_counter//(interval+1))
                    path = os.path.join(out_img_path, imgname)
                    cv2.imwrite(path, frame)
        cap.release()

    for i in range(anvideo_mov_f):
        aframe_counter = 0
        cap = cv2.VideoCapture(avideo_mov_f[i])
        if cap.isOpened():
            while True:
                ret  = cap.grab()
                if not ret:
                    break
                aframe_counter += 1
                if aframe_counter % (interval+1) == 0:
                    if frames_per_video is not None:
                        if aframe_counter // (interval+1) > frames_per_video:
                            break
                    ret , frame = cap.retrieve()
                    if frame is None:
                        break
                    if obj.findall(avideo_mov_f[i])[0][1].startswith('0'):
                        if obj.findall(avideo_mov_f[i])[0][1][1:].startswith('0'):
                            id = obj.findall(avideo_mov_f[i])[0][1][2:]
                        else:
                            id = obj.findall(avideo_mov_f[i])[0][1][1:]
                    else:
                        id = obj.findall(avideo_mov_f[i])[0][1]
                    imgname = "{0}_{1}_{2}_{3}_{4}.jpg".format('I',id,'fattack',i+anvideo_mov_h,aframe_counter//(interval+1))
                    path = os.path.join(out_img_path, imgname)
                    cv2.imwrite(path, frame)
        cap.release()
    print('Successfully !')

'''Split M'''
def video2img_M(input_path, train_output_root, test_output_root, train_list, test_list, interval, frames_per_video):
    '''
    :param input_path: video path which need to be split
    :param outpath:  splited img path
    :param interval: sample interval
    :return: None
    '''
    rpath = os.path.join(input_path, 'real')
    apath = os.path.join(input_path, 'attack')
    with open(train_list) as f:
        train_id = f.read()
        tr_id = train_id.split('\n')
    with open(test_list) as f:
        test_id = f.read()
        ts_id = test_id.split('\n')

    # .avi | .mov | .mp4
    rvideo_mov = sorted(glob(os.path.join(rpath, "*.mov")),key = os.path.getctime) # return a list of path
    rvideo_avi = sorted(glob(os.path.join(rpath, "*.avi")),key = os.path.getctime)  # return a list of path
    rvideo_mp4 = sorted(glob(os.path.join(rpath, "*.mp4")),key = os.path.getctime)  # return a list of path
    rvideo = rvideo_mov + rvideo_avi + rvideo_mp4
    rnvideo = len(rvideo)

    avideo_mov = glob(os.path.join(apath, "*.mov"))  # return a list of path
    avideo_avi = glob(os.path.join(apath, "*.avi"))  # return a list of path
    avideo_mp4 = glob(os.path.join(apath, "*.mp4"))  # return a list of path
    avideo = avideo_mov + avideo_avi + avideo_mp4
    anvideo = len(avideo)

    if not os.path.isdir(train_output_root):
        os.makedirs(train_output_root)
    else:
        shutil.rmtree(train_output_root)
        os.makedirs(train_output_root)
    if not os.path.isdir(test_output_root):
        os.makedirs(test_output_root)
    else:
        shutil.rmtree(test_output_root)
        os.makedirs(test_output_root)

    pattern = r"(\D+)(\d+)"
    obj = re.compile(pattern)

    for i in range(rnvideo):
        train_rframe_counter = 0
        val_rframe_counter = 0
        cap = cv2.VideoCapture(rvideo[i])
        print('real->',rvideo[i])
        if cap.isOpened():
            while True:
                ret,frame = cap.read()
                if not ret:
                    break
                if frame is None:
                    break
                # print(obj.findall(rvideo[i])[0][1][1:])
                if obj.findall(rvideo[i])[0][1][1:] in tr_id:
                    train_rframe_counter += 1
                    if train_rframe_counter % (interval + 1) == 0:
                        if frames_per_video is not None:
                            if train_rframe_counter // (interval + 1) > frames_per_video:
                                break
                        if obj.findall(rvideo[i])[0][1][1:].startswith('0'):
                            id = obj.findall(rvideo[i])[0][1][2:]
                        else:
                            id = obj.findall(rvideo[i])[0][1][1:]
                        imgname = "{0}_{1}_{2}_{3}_{4}.jpg".format('M', id, 'real',i, train_rframe_counter // (interval + 1))
                        path = os.path.join(train_output_root, imgname)
                        cv2.imwrite(path, frame)
                elif obj.findall(rvideo[i])[0][1][1:] in ts_id:
                    val_rframe_counter += 1
                    if val_rframe_counter % (interval + 1) == 0:
                        if frames_per_video is not None:
                            if val_rframe_counter // (interval+1) > frames_per_video:
                                break
                        if obj.findall(rvideo[i])[0][1][1:].startswith('0'):
                            id = obj.findall(rvideo[i])[0][1][2:]
                        else:
                            id = obj.findall(rvideo[i])[0][1][1:]
                        imgname = "{0}_{1}_{2}_{3}_{4}.jpg".format('M', id, 'real',i, val_rframe_counter // (interval + 1))
                        path = os.path.join(test_output_root, imgname)
                        cv2.imwrite(path, frame)
        cap.release()

    for i in range(anvideo):
        train_aframe_counter = 0
        val_aframe_counter = 0
        cap = cv2.VideoCapture(avideo[i])
        print('attack->',avideo[i])
        if cap.isOpened():
            while True:
                ret,frame = cap.read()
                if not ret:
                    break
                if frame is None:
                    break
                if obj.findall(avideo[i])[0][1][1:] in tr_id:
                    train_aframe_counter += 1
                    if train_aframe_counter % (interval + 1) == 0:
                        if frames_per_video is not None:
                            if train_aframe_counter // (interval+1) > frames_per_video:
                                break
                        if obj.findall(avideo[i])[0][1][1:].startswith('0'):
                            id = obj.findall(avideo[i])[0][1][2:]
                        else:
                            id = obj.findall(avideo[i])[0][1][1:]
                        imgname = "{0}_{1}_{2}_{3}_{4}.jpg".format('M', id, 'attack', i, train_aframe_counter // (interval + 1))
                        path = os.path.join(train_output_root, imgname)
                        cv2.imwrite(path, frame)
                elif obj.findall(avideo[i])[0][1][1:] in ts_id:
                    val_aframe_counter += 1
                    if val_aframe_counter % (interval + 1) == 0:
                        if frames_per_video is not None:
                            if val_aframe_counter // (interval+1) > frames_per_video:
                                break
                        if obj.findall(avideo[i])[0][1][1:].startswith('0'):
                            id = obj.findall(avideo[i])[0][1][2:]
                        else:
                            id = obj.findall(avideo[i])[0][1][1:]
                        imgname = "{0}_{1}_{2}_{3}_{4}.jpg".format('M', id, 'attack', i,val_aframe_counter // (interval + 1))
                        path = os.path.join(test_output_root, imgname)
                        cv2.imwrite(path, frame)
        cap.release()
    print('Successfully !')



'''CASIA-FASD'''
train_path = '/home/yjy/Dataset/CASIA_FASD/train_release'
test_path = '/home/yjy/Dataset/CASIA_FASD/test_release'
outpath = '/home/yjy/Dataset/OCMI/CASIA_FASD'
interval = 2
frames_per_video = 50
video2img_C(train_path,outpath,interval,frames_per_video,True)
video2img_C(test_path,outpath,interval,frames_per_video,False)

'''OULU'''
train_path = '/home/yjy/Dataset/Oulu_NPU/Train_files'
test_path = '/home/yjy/Dataset/Oulu_NPU/Test_files'

out_dir = '/home/yjy/Dataset/OCMI/Oulu_NPU'
interval = 2
frames_per_video = 50
video2img_O(train_path, out_dir, interval, frames_per_video, True)
video2img_O(test_path, out_dir, interval, frames_per_video, False)

'''Replay-Attack'''
train_path = '/home/yjy/Dataset/Replay_Attack/train'
test_path = '/home/yjy/Dataset/Replay_Attack/test'

out_dir_I = '/home/yjy/Dataset/OCMI/Replay_Attack'
interval = 0
frames_per_video = None
video2img_I(train_path, out_dir_I, interval, frames_per_video, True)
video2img_I(test_path, out_dir_I, interval, frames_per_video, False)



'''MSU-MFSD'''
# has train_sub_list and test_sub_list
root = '/home/yjy/Dataset/MSU_MFSD'
train_output_root = '/home/yjy/Dataset/OCMI/MSU_MFSD/train_img_MSU_MFSD'
test_output_root = '/home/yjy/Dataset/OCMI/MSU_MFSD/val_img_MSU_MFSD'
interval = 0
frames_per_video = None
video2img_M(root, train_output_root, test_output_root, root+r'/train_sub_list.txt', root+r'/test_sub_list.txt', interval, frames_per_video)



