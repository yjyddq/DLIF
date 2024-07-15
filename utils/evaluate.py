import os
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from utils.utils import AverageMeter, accuracy
from utils.statistic import get_EER_states, get_HTER_at_thr, calculate, calculate_threshold



def get_err_threhold(fpr, tpr, threshold):
    # tpr = tp / (tp + fn)
    # fpr = fp / (fp + tn)
    differ_tpr_fpr_1 = tpr + fpr - 1.0
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]
    return err, best_th, right_index

def performances_val(map_score_val_filename):
    '''
    Used to record the probability of liveness of test samples
    '''
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        try:
            count += 1
            tokens = line.split()
            score = float(tokens[0][7:])
            label = float(tokens[1][6:])
            id = tokens[2][3:]
            val_scores.append(score)
            val_labels.append(label)
            data.append({'map_score': score, 'label': label,'id':id})
            if label == 1:
                num_real += 1
            else:
                num_fake += 1
        except:
            continue
    fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
    auc_test = auc(fpr, tpr)
    val_err, val_threshold, right_index = get_err_threhold(fpr, tpr, threshold)

    type1 = len([s for s in data if s['map_score'] < val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    val_ACC = 1 - (type1 + type2) / count

    FRR = 1 - tpr  # FRR = 1 - TPR

    HTER = (fpr + FRR) / 2.0  # error recognition rate &  reject recognition rate

    return val_ACC, fpr[right_index], FRR[right_index], HTER[right_index], auc_test, val_err

def eval(valid_dataloader, u_encoder, classifier, criterion, config, epoch, local_rank):
    valid_losses = AverageMeter()
    valid_top1 = AverageMeter()
    prob_dict = {}
    label_dict = {}
    u_encoder.eval()
    classifier.eval()
    output_dict_tmp = {}
    target_dict_tmp = {}
    scores_list = []
    with torch.no_grad():
        for iter, (input, target, videoID) in enumerate(valid_dataloader):
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()

            fu = u_encoder(input,train=False,norm_flag=True)
            cls_out = classifier(fu,norm_flag=True)

            prob = F.softmax(config.s * cls_out, dim=1).cpu().data.numpy()[:, 1]
            label = target.cpu().data.numpy()
            for ii in range(input.shape[0]):  # batch size
                scores_list.append("scores:{} label:{} id:{}\n".format(prob[ii], label[ii], videoID[ii]))

            for i in range(len(prob)):
                if(videoID[i] in prob_dict.keys()):    # for example the videoID can be a string
                    prob_dict[videoID[i]].append(prob[i])  # these codes acheive the statistic of test videos' score
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                else:
                    prob_dict[videoID[i]] = []
                    label_dict[videoID[i]] = []
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]] = []
                    target_dict_tmp[videoID[i]] = []
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))

    if local_rank == 0:
        # Log the probability of liveness of test samples
        score_dir = config.logs + 'epoch_{}'.format(epoch)
        if not os.path.exists(score_dir):
            os.makedirs(score_dir)
        log_prob_filename = score_dir + "/{}_score.txt".format(config.protocol)
        with open(log_prob_filename, 'w') as file:
            file.writelines(scores_list)
    else:
        log_prob_filename = None

    prob_list = []
    label_list = []
    for key in prob_dict.keys():
        avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
        avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
        prob_list = np.append(prob_list, avg_single_video_prob)
        label_list = np.append(label_list, avg_single_video_label)
        # compute loss and acc for every video
        avg_single_video_output = sum(output_dict_tmp[key]) / len(output_dict_tmp[key])
        avg_single_video_target = sum(target_dict_tmp[key]) / len(target_dict_tmp[key])

        loss = criterion(avg_single_video_output, avg_single_video_target.long())
        acc_valid = accuracy(avg_single_video_output, avg_single_video_target, topk=(1,))
        valid_losses.update(loss.item())
        valid_top1.update(acc_valid[0])


    AUC = roc_auc_score(label_list, prob_list)
    cur_EER_valid, threshold, _, _ = get_EER_states(prob_list, label_list)
    ACC_threshold = calculate_threshold(prob_list, label_list, threshold)
    HTER = get_HTER_at_thr(prob_list, label_list, threshold)

    return [valid_losses.avg, valid_top1.avg, cur_EER_valid, HTER, AUC, threshold, ACC_threshold], log_prob_filename



