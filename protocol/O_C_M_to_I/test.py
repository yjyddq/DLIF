import sys
sys.path.append('../../')
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from config import config
from utils.evaluate import eval, performances_val
from dataset.dataset import FASDataset_val
from utils.utils import draw_roc,sample_frames_val
from sklearn.metrics import roc_curve, auc, roc_auc_score
from models.module import US_Encoder,Classifier
from utils.statistic import get_EER_states, get_HTER_at_thr, calculate, calculate_threshold

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus


def test(u_encoder, classifier, test_loader, score_root_path, name=""):
    prob_dict = {}
    label_dict = {}
    u_encoder.eval()
    classifier.eval()
    output_dict_tmp = {}
    target_dict_tmp = {}
    scores_list = []
    with torch.no_grad():
        for iter, (input, target, videoID) in enumerate(test_loader):
            input = input.cuda()
            target = torch.from_numpy(np.array(target)).long().cuda()

            fu = u_encoder(input, train=False, norm_flag=True)
            cls_out = classifier(fu, norm_flag=True)

            prob = F.softmax(config.s * cls_out, dim=1).cpu().data.numpy()[:, 1]
            label = target.cpu().data.numpy()
            for ii in range(input.shape[0]):  # batch size
                scores_list.append("scores:{} label:{} id:{}\n".format(prob[ii], label[ii], videoID[ii]))

            for i in range(len(prob)):
                if (videoID[i] in prob_dict.keys()):  # for example the videoID can be a string
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

    log_prob_filename = os.path.join(score_root_path, "{}_score.txt".format(name))

    print("score: write test scores to {}".format(log_prob_filename))
    with open(log_prob_filename, 'w') as file:
        file.writelines(scores_list)

    prob_list = []
    label_list = []
    for key in prob_dict.keys():
        avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
        avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
        prob_list = np.append(prob_list, avg_single_video_prob)
        label_list = np.append(label_list, avg_single_video_label)

    AUC = roc_auc_score(label_list, prob_list)
    cur_EER_valid, threshold, _, _ = get_EER_states(prob_list, label_list)
    ACC = calculate_threshold(prob_list, label_list, threshold)
    HTER = get_HTER_at_thr(prob_list, label_list, threshold)

    best_result_filename = os.path.join(score_root_path, "{}_best_performance.txt".format(name))
    with open(best_result_filename,'w') as file:
        file.writelines('Best AUC:{}'.format(AUC)+'\n'+
                        'Best HTER:{}'.format(HTER)+'\n'+
                        'Best ACC:{}'.format(ACC)+'\n')
    return HTER, AUC, ACC

def main():
    u_encoder = US_Encoder(config.model,config.pretrained)
    classifier = Classifier()
    u_encoder = nn.DataParallel(u_encoder).cuda()
    classifier = nn.DataParallel(classifier).cuda()
    tgt_test_data = sample_frames_val(dataset_name=config.tgt_data)
    test_dataloader = DataLoader(FASDataset_val(tgt_test_data),
                                 batch_size=256,num_workers=8,
                                 prefetch_factor=8, shuffle=True)
    print('\n')
    print("**Testing**")
    # load model
    u_encoder_ = torch.load(config.best_model_path + 'uencoder_best_17.pth.tar') # change to your best weight path
    classifier_ = torch.load(config.best_model_path + 'cls_best_17.pth.tar')
    u_encoder.load_state_dict(u_encoder_["state_dict"])
    classifier.load_state_dict(classifier_["state_dict"])
    # test model
    score_root_path = config.logs
    test_args = test(u_encoder, classifier, test_dataloader,  score_root_path, config.protocol)
    print('\n===========Test Info===========\n')
    print(config.tgt_data, 'Test HTER: %5.4f' %(test_args[0]))
    print(config.tgt_data, 'Test AUC: %5.4f' % (test_args[1]))
    print(config.tgt_data, 'Test ACC of threshold: %5.4f' % (test_args[2]))
    print('\n===============================\n')

if __name__ == '__main__':
    main()
