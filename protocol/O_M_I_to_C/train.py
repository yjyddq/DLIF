import sys

sys.path.append('../../')
import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from datetime import datetime
from timeit import default_timer as timer
from torch.nn.parallel import DistributedDataParallel as DDP

from config import config
from utils.evaluate import eval, performances_val
from loss.hard_triplet_loss import HardTripletLoss
from loss.losses import OrthogonalLoss, AMSoftmaxLoss, SupConLoss
from dataset.get_loader import get_dataset, get_dataset_train
from models.module import US_Encoder, VS_Encoder, Classifier, Discriminator
from utils.utils import save_checkpoint, AverageMeter, Logger, accuracy, mkdirs, adjust_learning_rate, time_to_str, \
    adjust_learning_rate_restart

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.cuda.manual_seed(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def train():
    # init the multiprocess and dist
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    # mkdirs
    if local_rank == 0:
        mkdirs(config.checkpoint_path, config.best_model_path, config.logs)

    # load data
    src1_train_dataloader, src2_train_dataloader, src3_train_dataloader, tgt_valid_dataloader = \
        get_dataset_train(config.protocol, config.src1_data, config.src2_data, config.src3_data, config.tgt_data,
                          config.batch_size, config.test_batch_size)

    # 0:loss, 1:top-1, 2:EER, 3:HTER, 4:ACER, 5:AUC, 6:ACC of threshold
    valid_args = [np.inf, 0, 0, 0, 0, 0, 0, 0]
    best_model_ACC = 0.0
    best_model_HTER = 1.0
    best_model_AUC = 0.0
    best_model_threshold = 0.5

    loss_classifier = AverageMeter()
    loss_identity = AverageMeter()
    loss_orthogonal = AverageMeter()
    loss_aaicu = AverageMeter()
    loss_aaicv = AverageMeter()
    classifer_top1 = AverageMeter()

    # build model
    u_encoder = US_Encoder(config.model, config.pretrained, src_num='three')
    v_encoder = VS_Encoder(config.model, config.pretrained, src_num='three')
    classifier = Classifier()
    discriminator = Discriminator(config.global_ID_num)
    u_encoder = DDP(u_encoder.cuda(), device_ids=[local_rank], output_device=local_rank)
    v_encoder = DDP(v_encoder.cuda(), device_ids=[local_rank], output_device=local_rank)
    classifier = DDP(classifier.cuda(), device_ids=[local_rank], output_device=local_rank)
    discriminator = DDP(discriminator.cuda(), device_ids=[local_rank], output_device=local_rank)

    # log
    if local_rank == 0:
        log = Logger()
        log.open(config.logs + config.tgt_data.split('/')[-1] + '_log.txt', mode='a')
        log.write("\n-------------------------------------------------------------------- [START %s] %s\n\n" % (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 67))
        log.write('** start training target model! **\n')
        log.write(
            '---------|------------- VALID -------------|--- classifier ---|--  aaicu  --|-- identity --|-- orthogonal --|--  aaicv  --|------ Current Best ------|--------------|-- lr --|\n')
        log.write(
            '  epoch  |   loss    ACC    HTER    AUC    |   loss   top-1   |     loss    |     loss     |      loss      |     loss    |    ACC   HTER    AUC     |     time     |\n')
        log.write(
            '--------------------------------------------------------------------------------------------------------------------------------------------------------------------|\n')
        start = timer()

    criterion = {
        'softmax': nn.CrossEntropyLoss().cuda(),
        'orthogonal': OrthogonalLoss().cuda(),
        'triplet': HardTripletLoss(margin=0.1, hardest=False).cuda(),
        'mse': nn.MSELoss().cuda(),
        'amsoftmax': AMSoftmaxLoss().cuda(),
        'supcontra': SupConLoss().cuda(),
    }
    optimizer_u_dict = [
        {"params": filter(lambda p: p.requires_grad, u_encoder.parameters()), "lr": config.init_lr},
        {"params": filter(lambda p: p.requires_grad, classifier.parameters()), "lr": config.init_lr}
    ]
    optimizer_v_dict = [
        {"params": filter(lambda p: p.requires_grad, v_encoder.parameters()), "lr": config.init_lr},
        {"params": filter(lambda p: p.requires_grad, discriminator.parameters()), "lr": config.init_lr}
    ]

    if config.optim == 'SGD':
        optimizer_u = optim.SGD(optimizer_u_dict, lr=config.init_lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)
        optimizer_v = optim.SGD(optimizer_v_dict, lr=config.init_lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)
    elif config.optim == 'Adam':
        optimizer_u = optim.Adam(optimizer_u_dict, lr=config.init_lr, weight_decay=config.weight_decay)
        optimizer_v = optim.Adam(optimizer_v_dict, lr=config.init_lr, weight_decay=config.weight_decay)
    elif config.optim == 'AdamW':
        optimizer_u = optim.AdamW(optimizer_u_dict, lr=config.init_lr, weight_decay=config.weight_decay)
        optimizer_v = optim.AdamW(optimizer_v_dict, lr=config.init_lr, weight_decay=config.weight_decay)
    init_param_lr_u = []
    init_param_lr_v = []

    for param_group in optimizer_u.param_groups:
        init_param_lr_u.append(param_group["lr"])
    for param_group in optimizer_v.param_groups:
        init_param_lr_v.append(param_group["lr"])

    src1_train_iter = iter(src1_train_dataloader)
    src1_iter_per_epoch = len(src1_train_iter)
    src2_train_iter = iter(src2_train_dataloader)
    src2_iter_per_epoch = len(src2_train_iter)
    src3_train_iter = iter(src3_train_dataloader)
    src3_iter_per_epoch = len(src3_train_iter)

    iter_per_epoch = config.iter_per_epoch
    max_iter = config.max_iter
    epoch = 1

    # train
    for iter_num in range(1, max_iter + 1):
        u_encoder.train(True)
        v_encoder.train(True)
        classifier.train(True)
        discriminator.train(True)

        param_lr_tmp_u = []
        for param_group in optimizer_u.param_groups:
            param_lr_tmp_u.append(param_group["lr"])

        optimizer_u.zero_grad()
        optimizer_v.zero_grad()

        ######### data prepare #########
        src1_img_real, src1_label_real, src1_id_real, src1_img_fake, src1_label_fake, src1_id_fake = src1_train_iter.next()
        src1_img_real = src1_img_real.squeeze(0).cuda()
        src1_label_real = src1_label_real.squeeze(0).cuda()
        src1_id_real = src1_id_real.squeeze(0).cuda()
        src1_img_fake = src1_img_fake.squeeze(0).cuda().squeeze(0)
        src1_label_fake = src1_label_fake.squeeze(0).cuda()
        src1_id_fake = src1_id_fake.squeeze(0).cuda()
        input1_real_shape = src1_img_real.shape[0]
        input1_fake_shape = src1_img_fake.shape[0]

        src2_img_real, src2_label_real, src2_id_real, src2_img_fake, src2_label_fake, src2_id_fake = src2_train_iter.next()
        src2_img_real = src2_img_real.squeeze(0).cuda()
        src2_label_real = src2_label_real.squeeze(0).cuda()
        src2_id_real = src2_id_real.squeeze(0).cuda()
        src2_img_fake = src2_img_fake.squeeze(0).cuda()
        src2_label_fake = src2_label_fake.squeeze(0).cuda()
        src2_id_fake = src2_id_fake.squeeze(0).cuda()
        input2_real_shape = src2_img_real.shape[0]
        input2_fake_shape = src2_img_fake.shape[0]

        src3_img_real, src3_label_real, src3_id_real, src3_img_fake, src3_label_fake, src3_id_fake = src3_train_iter.next()
        src3_img_real = src3_img_real.squeeze(0).cuda()
        src3_label_real = src3_label_real.squeeze(0).cuda()
        src3_id_real = src3_id_real.squeeze(0).cuda()
        src3_img_fake = src3_img_fake.squeeze(0).cuda()
        src3_label_fake = src3_label_fake.squeeze(0).cuda()
        src3_id_fake = src3_id_fake.squeeze(0).cuda()
        input3_real_shape = src3_img_real.shape[0]
        input3_fake_shape = src3_img_fake.shape[0]

        input_data = torch.cat([src1_img_real, src2_img_real, src3_img_real,
                                src1_img_fake, src2_img_fake, src3_img_fake], dim=0)

        source_label = torch.cat([src1_label_real, src2_label_real, src3_label_real,
                                  src1_label_fake, src2_label_fake, src3_label_fake], dim=0)

        source_id_label = torch.cat([src1_id_real, src2_id_real, src3_id_real,
                                     src1_id_fake, src2_id_fake, src3_id_fake], dim=0)

        ######### forward #########
        fu_q_norm, fu_k_normH, fu_k_normM, fu_k_normL = u_encoder(input_data, train=True, norm_flag=True)
        fv_q_norm, fv_k_normH, fv_k_normM, fv_k_normL = v_encoder(input_data, train=True, norm_flag=True)
        cls_out = classifier(fu_q_norm, norm_flag=True)
        id_out = discriminator(fv_q_norm, norm_flag=True)

        ######### ortho loss #########
        # The augmented instances involved are adjustable, depending on different Aug Flows
        ortho_loss_u = criterion['orthogonal'](torch.cat([fu_q_norm, fu_k_normH, fu_k_normL], dim=0),
                                               torch.cat([fv_q_norm, fv_k_normH, fv_k_normL], dim=0).detach())
        ortho_loss_v = criterion['orthogonal'](torch.cat([fu_q_norm, fu_k_normH, fu_k_normL], dim=0).detach(),
                                               torch.cat([fv_q_norm, fv_k_normH, fv_k_normL], dim=0))

        ######### contra loss #########
        # The augmented instances involved are adjustable, depending on different Aug Flows
        # for the real samples, we assign the same label both org and aug
        real_cluster_label_1 = torch.LongTensor(input1_real_shape, 1).fill_(0).cuda()
        real_cluster_label_2 = torch.LongTensor(input2_real_shape, 1).fill_(0).cuda()
        real_cluster_label_3 = torch.LongTensor(input3_real_shape, 1).fill_(0).cuda()

        # for the attack samples we assign unique label for each istance in a batch, the aug share a copy from org
        # We generate a sequence of sequential integers as long as the attack samples as their labels, due to real samples occupy 0, the number of attack samples starts from 1
        fake_cluster_label_1 = torch.arange(1, input1_fake_shape + 1).view(-1, 1).cuda()
        fake_cluster_label_2 = torch.arange(input1_fake_shape + 1, input1_fake_shape + input2_fake_shape + 1).view(-1,
                                                                                                                   1).cuda()
        fake_cluster_label_3 = torch.arange(input1_fake_shape + input2_fake_shape + 1,
                                            input1_fake_shape + input2_fake_shape + input3_fake_shape + 1).view(-1,
                                                                                                                1).cuda()
        source_cluster_label = torch.cat([real_cluster_label_1, real_cluster_label_2, real_cluster_label_3,
                                          fake_cluster_label_1, fake_cluster_label_2, fake_cluster_label_3],
                                         dim=0).view(-1)

        aaic_u_loss = criterion["supcontra"](
            torch.cat([fu_q_norm.unsqueeze(1), fu_k_normH.unsqueeze(1), fu_k_normL.unsqueeze(1)], dim=1),
            source_cluster_label)
        aaic_v_loss = criterion["supcontra"](
            torch.cat([fv_q_norm.unsqueeze(1), fv_k_normH.unsqueeze(1), fv_k_normL.unsqueeze(1)], dim=1),
            source_id_label)

        ######### id loss #########
        id_loss = criterion["softmax"](config.s * id_out, source_id_label)

        id_amb = discriminator(fu_q_norm, norm_flag=True)
        id_amb_label = torch.FloatTensor(input_data.size(0), config.global_ID_num).fill_(
            1 / config.global_ID_num).cuda()
        id_amb_loss = criterion["mse"](id_amb, id_amb_label)

        ######### cls loss #########
        cls_loss = criterion["amsoftmax"](cls_out.narrow(0, 0, input_data.size(0)), source_label)

        ######### backward #########
        FAS_loss = cls_loss + config.lambda_aaicu * aaic_u_loss + config.lambda_ortho * ortho_loss_u + config.lambda_amb * id_amb_loss
        FAS_loss.backward(retain_graph=True)
        optimizer_u.step()
        optimizer_u.zero_grad()
        optimizer_v.zero_grad(set_to_none=True)

        li_amb = classifier(fv_q_norm, norm_flag=True)
        li_amb_label = torch.FloatTensor(input_data.size(0), 2).fill_(0.5).cuda()
        li_amb_loss = criterion["mse"](li_amb, li_amb_label)
        IDD_loss = id_loss + config.lambda_aaicv * aaic_v_loss + config.lambda_ortho * ortho_loss_v + config.lambda_amb * li_amb_loss

        IDD_loss.backward()
        optimizer_v.step()
        optimizer_v.zero_grad()
        optimizer_u.zero_grad(set_to_none=True)

        loss_classifier.update(cls_loss.item())
        loss_orthogonal.update((ortho_loss_u + ortho_loss_v).item())
        loss_identity.update(id_loss.item())
        loss_aaicu.update(aaic_u_loss.item())
        loss_aaicv.update(aaic_v_loss.item())

        acc = accuracy(cls_out.narrow(0, 0, input_data.size(0)), source_label, topk=(1,))
        classifer_top1.update(acc[0])
        if local_rank == 0:
            print('\r', end='', flush=True)
            print(
                '  %4.1f   |  %5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |   %6.3f    |    %6.3f    |     %6.3f     |    %6.3f   |  %6.3f  %6.3f  %6.3f  | %s'
                % (
                    iter_num / iter_per_epoch,
                    valid_args[0], valid_args[6], valid_args[3] * 100, valid_args[4] * 100,
                    loss_classifier.avg, classifer_top1.avg, loss_aaicu.avg, loss_identity.avg, loss_orthogonal.avg,
                    loss_aaicv.avg,
                    float(best_model_ACC * 100), float(best_model_HTER * 100), float(best_model_AUC * 100),
                    time_to_str(timer() - start, 'min'))
                , end='', flush=True)

        # evaluate the model and save the weight of the best model
        if (iter_num % iter_per_epoch == 0 and local_rank == 0):
            # 0:loss, 1:top-1, 2:EER, 3:HTER, 4:AUC, 5:threshold, 6:ACC of threshold
            valid_args, log_prob_filename = eval(tgt_valid_dataloader, u_encoder, classifier, criterion["amsoftmax"],
                                                 config, epoch, local_rank)

            is_best = False
            if ((valid_args[4] - valid_args[3]) >= (best_model_AUC - best_model_HTER)):
                is_best = True
                best_model_HTER = valid_args[3]
                best_model_ACC = valid_args[6]
                best_model_AUC = valid_args[4]
                best_model_threshold = valid_args[5]

            test_ACC, fpr, FRR, HTER, auc_test, test_err = performances_val(log_prob_filename)
            final_result_file = config.logs + 'epoch_{}/'.format(epoch) + "{}_best_performance.txt".format(
                config.protocol)
            with open(final_result_file, 'w') as f:
                f.write('EPOCH:{}\nAUC:{}\nHTER:{}\n'.format(epoch, auc_test, HTER))

            save_list = [epoch, valid_args, best_model_HTER, best_model_ACC, best_model_AUC, best_model_threshold]
            save_checkpoint(save_list, is_best, u_encoder, config.gpus, config.checkpoint_path, config.best_model_path,
                            filename='uencoder')
            save_checkpoint(save_list, is_best, classifier, config.gpus, config.checkpoint_path, config.best_model_path,
                            filename='cls')
            save_checkpoint(save_list, is_best, v_encoder, config.gpus, config.checkpoint_path, config.best_model_path,
                            filename='vencoder')
            save_checkpoint(save_list, is_best, discriminator, config.gpus, config.checkpoint_path,
                            config.best_model_path, filename='dis')
            print('\r', end='', flush=True)
            log.write(
                '  %4.1f   |  %5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |   %6.3f    |    %6.3f     |     %6.3f     |   %6.3f   |  %6.3f  %6.3f  %6.3f  | %s | %f'
                % (
                    iter_num / iter_per_epoch,
                    valid_args[0], valid_args[6], valid_args[3] * 100, valid_args[4] * 100,
                    loss_classifier.avg, classifer_top1.avg, loss_aaicu.avg, loss_identity.avg, loss_orthogonal.avg,
                    loss_aaicv.avg,
                    float(best_model_ACC * 100), float(best_model_HTER * 100), float(best_model_AUC * 100),
                    time_to_str(timer() - start, 'min'),
                    param_lr_tmp_u[0]))
            log.write('\n')
            time.sleep(0.01)

        # learning rate schedular
        adjust_learning_rate(optimizer_u, epoch, init_param_lr_u, config.final_lr, config.step, config.gamma,
                             config.warmup_epochs)
        adjust_learning_rate(optimizer_v, epoch, init_param_lr_v, config.final_lr, config.step, config.gamma,
                             config.warmup_epochs)

        # If you reach the end of the Iterator,then reload the dataset
        if (iter_num % src1_iter_per_epoch == 0):
            src1_train_dataloader.sampler.set_epoch(iter_num // src1_iter_per_epoch)
            src1_train_iter = iter(src1_train_dataloader)
        if (iter_num % src2_iter_per_epoch == 0):
            src2_train_dataloader.sampler.set_epoch(iter_num // src2_iter_per_epoch)
            src2_train_iter = iter(src2_train_dataloader)
        if (iter_num % src3_iter_per_epoch == 0):
            src3_train_dataloader.sampler.set_epoch(iter_num // src3_iter_per_epoch)
            src3_train_iter = iter(src3_train_dataloader)
        if (iter_num % iter_per_epoch == 0):
            epoch = epoch + 1


if __name__ == '__main__':
    train()




















