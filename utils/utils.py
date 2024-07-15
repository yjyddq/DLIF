import os
import sys
import json
import math
import torch
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Intra Dataset ID token
SOURCE_ID_BANK = {'Oulu_NPU':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                   'CASIA_FASD':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                   'MSU_MFSD':[2,3,5,6,7,8,9,11,12,21,22,34,53,54,55],
                   'Replay_Attack':[1,2,4,6,7,8,12,16,18,25,27,103,105,108,110]}

# Assign a global ID token to each subject from different datasets
global_ID_table = {'O_C_M_to_I':
                       {'O_1': 0, 'O_2': 1, 'O_3': 2, 'O_4': 3, 'O_5': 4, 'O_6': 5, 'O_7': 6, 'O_8': 7, 'O_9': 8, 'O_10': 9,
                   'O_11': 10, 'O_12': 11, 'O_13': 12, 'O_14': 13, 'O_15': 14, 'O_16': 15, 'O_17': 16, 'O_18': 17, 'O_19': 18, 'O_20': 19,
                   'C_1': 20, 'C_2': 21, 'C_3': 22, 'C_4': 23, 'C_5': 24, 'C_6': 25, 'C_7': 26, 'C_8': 27, 'C_9': 28,'C_10': 29,
                   'C_11': 30, 'C_12': 31, 'C_13': 32, 'C_14': 33, 'C_15': 34, 'C_16': 35, 'C_17': 36, 'C_18': 37,'C_19': 38, 'C_20': 39,
                   'M_2': 40, 'M_3': 41, 'M_5': 42, 'M_6': 43, 'M_7': 44, 'M_8': 45, 'M_9': 46, 'M_11': 47, 'M_12': 48, 'M_21': 49,
                   'M_22': 50, 'M_34': 51, 'M_53': 52, 'M_54': 53, 'M_55': 54},

                   'O_C_I_to_M':
                       {'O_1': 0, 'O_2': 1, 'O_3': 2, 'O_4': 3, 'O_5': 4, 'O_6': 5, 'O_7': 6, 'O_8': 7, 'O_9': 8, 'O_10': 9,
                   'O_11': 10, 'O_12': 11, 'O_13': 12, 'O_14': 13, 'O_15': 14, 'O_16': 15, 'O_17': 16, 'O_18': 17, 'O_19': 18, 'O_20': 19,
                   'C_1': 20, 'C_2': 21, 'C_3': 22, 'C_4': 23, 'C_5': 24, 'C_6': 25, 'C_7': 26, 'C_8': 27, 'C_9': 28,'C_10': 29,
                   'C_11': 30, 'C_12': 31, 'C_13': 32, 'C_14': 33, 'C_15': 34, 'C_16': 35, 'C_17': 36, 'C_18': 37,'C_19': 38, 'C_20': 39,
                   'I_1': 40, 'I_2': 41, 'I_4': 42, 'I_6': 43, 'I_7': 44, 'I_8': 45, 'I_12': 46, 'I_16': 47, 'I_18': 48, 'I_25': 49,
                   'I_27': 50, 'I_103': 51, 'I_105': 52, 'I_108': 53, 'I_110': 54},

                   'O_M_I_to_C':
                       {'O_1': 0, 'O_2': 1, 'O_3': 2, 'O_4': 3, 'O_5': 4, 'O_6': 5, 'O_7': 6, 'O_8': 7, 'O_9': 8, 'O_10': 9,
                   'O_11': 10, 'O_12': 11, 'O_13': 12, 'O_14': 13, 'O_15': 14, 'O_16': 15, 'O_17': 16, 'O_18': 17, 'O_19': 18, 'O_20': 19,
                   'M_2': 20, 'M_3': 21, 'M_5': 22, 'M_6': 23, 'M_7': 24, 'M_8': 25, 'M_9': 26, 'M_11': 27, 'M_12': 28, 'M_21': 29,
                   'M_22': 30, 'M_34': 31, 'M_53': 32, 'M_54': 33, 'M_55': 34,
                   'I_1': 35, 'I_2': 36, 'I_4': 37, 'I_6': 38, 'I_7': 39, 'I_8': 40, 'I_12': 41, 'I_16': 42, 'I_18': 43, 'I_25': 44,
                   'I_27': 45, 'I_103': 46, 'I_105': 47, 'I_108': 48, 'I_110': 49},

                   'I_C_M_to_O':
                       {'I_1': 0, 'I_2': 1, 'I_4': 2, 'I_6': 3, 'I_7': 4, 'I_8': 5, 'I_12': 6, 'I_16': 7, 'I_18': 8, 'I_25': 9,
                    'I_27': 10, 'I_103': 11, 'I_105': 12, 'I_108': 13, 'I_110': 14,
                    'C_1': 15, 'C_2': 16, 'C_3': 17, 'C_4': 18, 'C_5': 19, 'C_6': 20, 'C_7': 21, 'C_8': 22,'C_9': 23, 'C_10': 24,
                    'C_11': 25, 'C_12': 26, 'C_13': 27, 'C_14': 28, 'C_15': 29, 'C_16': 30, 'C_17': 31, 'C_18': 32,'C_19': 33, 'C_20': 34,
                    'M_2': 35, 'M_3': 36, 'M_5': 37, 'M_6': 38, 'M_7': 39, 'M_8': 40, 'M_9': 41, 'M_11': 42,'M_12': 43, 'M_21': 44,
                    'M_22': 45, 'M_34': 46, 'M_53': 47, 'M_54': 48, 'M_55': 49},

                   'M_I_to_C':
                       {'M_2': 0, 'M_3': 1, 'M_5': 2, 'M_6': 3, 'M_7': 4, 'M_8': 5, 'M_9': 6, 'M_11': 7, 'M_12': 8, 'M_21': 9,
                   'M_22': 10, 'M_34': 11, 'M_53': 12, 'M_54': 13, 'M_55': 14,
                   'I_1': 15, 'I_2': 16, 'I_4': 17, 'I_6': 18, 'I_7': 19, 'I_8': 20, 'I_12': 21, 'I_16': 22, 'I_18': 23, 'I_25': 24,
                   'I_27': 25, 'I_103': 26, 'I_105': 27, 'I_108': 28, 'I_110': 29},
                   'M_I_to_O':
                       {'M_2': 0, 'M_3': 1, 'M_5': 2, 'M_6': 3, 'M_7': 4, 'M_8': 5, 'M_9': 6, 'M_11': 7, 'M_12': 8, 'M_21': 9,
                   'M_22': 10, 'M_34': 11, 'M_53': 12, 'M_54': 13, 'M_55': 14,
                   'I_1': 15, 'I_2': 16, 'I_4': 17, 'I_6': 18, 'I_7': 19, 'I_8': 20, 'I_12': 21, 'I_16': 22, 'I_18': 23, 'I_25': 24,
                   'I_27': 25, 'I_103': 26, 'I_105': 27, 'I_108': 28, 'I_110': 29},
                   }

def adjust_learning_rate_restart(optimizer, epoch, init_param_lr, final_param_lr, epochs, warmup_epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch <= warmup_epochs:
        i = 0
        for param_group in optimizer.param_groups:
            init_lr = init_param_lr[i]
            i += 1
            param_group['lr'] = init_lr * epoch / warmup_epochs
    else:
        i = 0
        for param_group in optimizer.param_groups:
            init_lr = init_param_lr[i]
            i += 1
            lr = final_param_lr + (init_lr - final_param_lr) * 0.5 * \
                 (1. + math.cos(math.pi * epoch / epochs))
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr

def adjust_learning_rate(optimizer, epoch, init_param_lr, final_param_lr, step, gamma, warmup_epochs):

    if epoch <= warmup_epochs:
        i = 0
        for param_group in optimizer.param_groups:
            init_lr = init_param_lr[i]
            i += 1
            param_group['lr'] = init_lr * epoch / warmup_epochs
    else:
        i = 0
        for param_group in optimizer.param_groups:
            init_lr = init_param_lr[i]
            i += 1
            if(epoch % step ==0):
                if init_lr * gamma ** (epoch // step) < final_param_lr:
                    param_group['lr'] = final_param_lr
                else:
                    param_group['lr'] = init_lr * gamma ** (epoch // step)


def draw_roc(frr_list, far_list, roc_auc):
    plt.switch_backend('agg')
    plt.rcParams['figure.figsize'] = (6.0, 6.0)
    plt.title('ROC')
    plt.plot(far_list, frr_list, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='upper right')
    plt.plot([0, 1], [1, 0], 'r--')
    plt.grid(ls='--')
    plt.ylabel('False Negative Rate')
    plt.xlabel('False Positive Rate')
    save_dir = './save_results/ROC/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig('./save_results/ROC/ROC.png')
    file = open('./save_results/ROC/FAR_FRR.txt', 'w')
    save_json = []
    dict = {}
    dict['FAR'] = far_list
    dict['FRR'] = frr_list
    save_json.append(dict)
    json.dump(save_json, file, indent=4)


'''My sample frames'''
def sample_frames_val(dataset_name):
    '''
        from every video (frames) to sample num_frames to test
        return: the choosen frames' path , label, ID, video ID
    '''
    # The process is a litter cumbersome, you can change to your way for convenience
    # use the target test dataset as the testset & valset
    root_path = dataset_name
    test_label_path = root_path + r'/val_label.csv'
    test_landmarks_frame = pd.read_csv(test_label_path, delimiter=",", header=None)
    test_img_path_list = test_landmarks_frame.iloc[:, 0].tolist()
    test_label_list = test_landmarks_frame.iloc[:, 1].tolist()
    subject_id_list = test_landmarks_frame.iloc[:, 2].tolist()
    video_name_list = []
    for i, path in enumerate(test_img_path_list):
        frame_id = path.split('/')[-1][:-4].split('_')
        video_id = frame_id[0] + '_' + frame_id[1] + '_' + frame_id[2] + '_' + frame_id[3]
        video_name_list.append(video_id)
    sample_data_pd = pd.DataFrame()
    sample_data_pd['photo_path'] = test_img_path_list
    sample_data_pd['photo_label'] = test_label_list
    sample_data_pd['photo_belong_to_video_ID'] = video_name_list
    sample_data_pd['subject_id'] = subject_id_list

    return sample_data_pd


def sample_frames_train(protocol,dataset_name):
    '''
        from every video (frames) to sample num_frames to test
        return: the choosen frames' path , label, ID, video ID
    '''
    # The process is a litter cumbersome, you can change to your way for convenience
    # use the source train dataset as the trainset
    root_path = dataset_name
    video_dict = {}

    label_path = root_path + r'/train_label.csv'
    landmarks_frame = pd.read_csv(label_path, delimiter=",", header=None)
    img_path_list = landmarks_frame.iloc[:, 0].tolist()
    label_list = landmarks_frame.iloc[:, 1].tolist()
    for i, path in enumerate(img_path_list):
        ret = path.find('real')
        if ret != -1:
            if ('real' not in video_dict.keys()):
                video_dict['real'] = {}
            img_name = path.split('/')[-1][:-4].split('_')
            id = int(img_name[1])
            frame_id = img_name[3] + '_' + img_name[4]
            if id in video_dict['real'].keys():
                video_dict['real'][id][frame_id] = {}
                video_dict['real'][id][frame_id]['img_path'] = path
                video_dict['real'][id][frame_id]['label'] = label_list[i]
                video_dict['real'][id][frame_id]['global_id'] = global_ID_table[protocol][img_name[0] + '_' + img_name[1]]
            else:
                video_dict['real'][id] = {}
                video_dict['real'][id][frame_id] = {}
                video_dict['real'][id][frame_id]['img_path'] = path
                video_dict['real'][id][frame_id]['label'] = label_list[i]
                video_dict['real'][id][frame_id]['global_id'] = global_ID_table[protocol][img_name[0] + '_' + img_name[1]]
        else:
            if ('attack' not in video_dict.keys()):
                video_dict['attack'] = {}
            img_name = path.split('/')[-1][:-4].split('_')
            id = int(img_name[1])
            frame_id = img_name[3] + '_' + img_name[4]
            if id in video_dict['attack'].keys():
                video_dict['attack'][id][frame_id] = {}
                video_dict['attack'][id][frame_id]['img_path'] = path
                video_dict['attack'][id][frame_id]['label'] = label_list[i]
                video_dict['attack'][id][frame_id]['global_id'] = global_ID_table[protocol][img_name[0] + '_' + img_name[1]]
            else:
                video_dict['attack'][id] = {}
                video_dict['attack'][id][frame_id] = {}
                video_dict['attack'][id][frame_id]['img_path'] = path
                video_dict['attack'][id][frame_id]['label'] = label_list[i]
                video_dict['attack'][id][frame_id]['global_id'] = global_ID_table[protocol][img_name[0] + '_' + img_name[1]]

    return video_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True) # return values, indices
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def mkdirs(checkpoint_path, best_model_path, logs):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    if not os.path.exists(logs):
        os.mkdir(logs)

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)
    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)
    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message:
            is_file = 0
        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def save_checkpoint(save_list, is_best, model, gpus, checkpoint_path, best_model_path, filename):
    epoch = save_list[0]
    valid_args = save_list[1]
    best_model_HTER = round(save_list[2], 5)
    best_model_ACC = save_list[3]
    best_model_AUC= save_list[4]
    best_model_threshold= save_list[5]
    if(len(gpus) > 1):
        old_state_dict = model.state_dict()
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in old_state_dict.items():
            flag = k.find('.module.')
            if (flag != -1):
                k = k.replace('.module.', '.')
            new_state_dict[k] = v
        state = {
            "epoch": epoch,
            "state_dict": new_state_dict,
            "valid_arg": valid_args,
            "best_model_HTER": best_model_HTER,
            "best_model_AUC": best_model_AUC,
            "best_model_ACC": best_model_ACC,
            "threshold": best_model_threshold
        }
    else:
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "valid_arg": valid_args,
            "best_model_HTER": best_model_HTER,
            "best_model_AUC": best_model_AUC,
            "best_model_ACC": best_model_ACC,
            "threshold": best_model_threshold
        }
    filepath = checkpoint_path + filename + '_checkpoint.pth.tar'
    torch.save(state, filepath)
    # just save best model
    if is_best:
        shutil.copy(filepath, best_model_path + filename + '_best_' + str(epoch) + '.pth.tar')

def zero_param_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()


if __name__ == '__main__':
    dataset_name = r'/home/yangjy/Dataset/OCMI/CASIA_FASD'
    sample_data_pd = sample_frames_val(dataset_name)
    print(sample_data_pd['photo_belong_to_video_ID'])
    print(len(sample_data_pd['photo_label']))