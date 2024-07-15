import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import ResNet, BasicBlock
from torch.autograd import Variable


'''Backbone'''
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    # change your path
    # model_path = r'/home/sist/yjy/myproj/MyModel_v3/pretrained_model/resnet18-5c106cde.pth'
    model_path = r'/home/yangjy/myproj/MyModel_v7/pretrained_model/resnet18-5c106cde.pth'
    if pretrained:
        model.load_state_dict(torch.load(model_path))
        print("loading model: ", model_path)
    return model


'''CWSA'''
class CWSA(nn.Module):
    """Modifying based on Squeeze-and-Excitation Networks
            refer to : https://github.com/huijie-frank/SENet
    """
    def __init__(self, channel, reduction=16):
        super(CWSA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(2*channel, 2*channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2*channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        mean = self.avg_pool(x).view(b, c)
        var = torch.var(x.view(b, c, -1),dim=-1,keepdim=True).view(b, c)
        y = torch.cat([mean,var],dim=-1)

        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


'''Style Cross'''
def calc_ins_mean_std(x, eps=1e-5):
    """extract feature map statistics"""
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = x.size()
    assert (len(size) == 4)
    N, C = size[:2]
    var = x.contiguous().view(N, C, -1).var(dim=2) + eps
    std = var.sqrt().view(N, C, 1, 1)
    mean = x.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return mean, std

def instance_norm_mix(content_feat, style_feat):
    """replace content statistics with style statistics"""
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_ins_mean_std(style_feat)
    content_mean, content_std = calc_ins_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def cn_rand_bbox(size, beta, bbx_thres):
    """sample a bounding box for cropping."""
    W = size[2]
    H = size[3]
    while True:
        ratio = np.random.beta(beta, beta)
        cut_rat = np.sqrt(ratio)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        ratio = float(bbx2 - bbx1) * (bby2 - bby1) / (W * H)
        if ratio > bbx_thres:
            break

    return bbx1, bby1, bbx2, bby2

def cn_op_2ins_space_chan(x, src_num='three', crop='neither', mode='domain',beta=1, bbx_thres=0.1, lam=None, chan=False):
    """
    It is worth noting that there are some advanced operations in CrossNorm,
    such as crossing styles between different channels, cropping bbx to cross styles,
    and whether the exchange are randomly performed or among domains.
    We did not discuss this in our paper, which is left for interested researchers to explore.
    Some details about CrossNorm can be referenced : https://arxiv.org/abs/2102.02811 (CrossNorm and SelfNorm for Generalization under Distribution Shifts)
    """
    assert crop in ['neither', 'style', 'content', 'both']
    assert mode in ['liveness', 'id']
    assert src_num in ['three', 'two']

    ins_idxs = torch.arange(len(x))
    if mode == 'liveness':
        if src_num == 'three':
            # there are two implementation methods in liveness mode [cross domain, random]
            p = torch.randint(0,1,(1,))
            if p == 1:
                ins_idxs = ins_idxs.roll(len(x)// 3,dims=0) # cross among different domains via roll
            else:
                ins_idxs = ins_idxs.roll(-len(x)//3,dims=0)
            # ins_idxs = torch.randperm(x.size()[0]).to(x.device) # cross randomly
        elif src_num == 'two':
            ins_idxs = ins_idxs.roll(len(x) // 2, dims=0)
            # ins_idxs = torch.randperm(x.size()[0]).to(x.device)

    if mode == 'id':
        ins_idxs = ins_idxs.roll(len(x) // 2, dims=0).to(x.device) # Exchange the styles of different liveness instances of the same identity via roll


    if crop in ['style', 'both']:
        bbx3, bby3, bbx4, bby4 = cn_rand_bbox(x.size(), beta=beta, bbx_thres=bbx_thres)
        x2 = x[ins_idxs, :, bbx3:bbx4, bby3:bby4]
    else:
        x2 = x[ins_idxs]

    if chan:
        chan_idxs = torch.randperm(x.size()[1]).to(x.device)

        x2 = x2[:, chan_idxs, :, :]

    if crop in ['content', 'both']:
        x_aug = torch.zeros_like(x)
        bbx1, bby1, bbx2, bby2 = cn_rand_bbox(x.size(), beta=beta, bbx_thres=bbx_thres)
        x_aug[:, :, bbx1:bbx2, bby1:bby2] = instance_norm_mix(content_feat=x[:, :, bbx1:bbx2, bby1:bby2],
                                                              style_feat=x2)

        mask = torch.ones_like(x, requires_grad=False)
        mask[:, :, bbx1:bbx2, bby1:bby2] = 0.
        x_aug = x * mask + x_aug
    else:
        x_aug = instance_norm_mix(content_feat=x, style_feat=x2)

    if lam is not None:
        x = x * lam + x_aug * (1-lam)
    else:
        x = x_aug

    return x

class StyleCross(nn.Module):
    """Modifying based on CrossNorm
        refer to : https://github.com/amazon-research/crossnorm-selfnorm
    """
    def __init__(self, src_num='three', crop=None, beta=None,chan=False):
        super(StyleCross, self).__init__()
        self.active = False
        self.cn_op = functools.partial(cn_op_2ins_space_chan,
                                       crop=crop, beta=beta,chan=chan)
        self.src_num = src_num
    def forward(self, x,mode):
        if mode == 'liveness':
            if self.active:
                n = x.shape[0] // 2
                src_real = x.narrow(0, 0, n)
                src_fake = x.narrow(0, n, n)
                src_real_k = self.cn_op(src_real,self.src_num,mode='liveness')
                src_fake_k = self.cn_op(src_fake,self.src_num,mode='liveness')
                invariant_liveness = torch.cat([src_real_k,src_fake_k],dim=0)

                self.active = False
                return invariant_liveness
        elif mode == 'id':
            if self.active:
                invariant_id = self.cn_op(x, self.src_num, mode='id')
                self.active = False
                return invariant_id

class SSA(nn.Module):
    '''
    For the sake of fairness, parameter layers are not used here either.
    idx is used to determine the contrast labels.
    If the liveness labels remain consistent before and after exchange,the labels remain unchanged.
    If the style label and content label are different, the liveness label uses the style label.
    '''
    def __init__(self):
        super(SSA, self).__init__()
        self.active = False
    def forward(self, x):
        if self.active:
            idx = torch.randint(len(x), (1,))
            x2 = x[idx]
            x_aug = instance_norm_mix(content_feat=x, style_feat=x2)

            self.active = False
            return x_aug, idx


'''Encoder'''
class US_Encoder_parallel(nn.Module):
    def __init__(self, model, pretrained, aug_code='1110', src_num='three'):
        '''
        :param model: resnet18
        :param pretrained: True | False
        :param aug_code:  The first digit of the binary code indicates mul(cascade 0) or add(parallel 1),
                            the remaining three represent whether a Style Cross occurred at the corresponding level, corresponding to L M H from low(right) to high(left), respectively
                            for example 1011 -> add None Middle Low, 0100 mul High None None
        add flow { '1011': L+M,'1101':L+H ,'1110':M+H, '1111':L+M+H } mul flow { '0001': L,'0010':M ,'0100':H, '0011':L*M, '0101':L*H, '0110':M*H, '0111':L*M*H }
        '0000' '1000' -> org
        '1001' L '1010' M '1100' H -> Repeat in mul flows, only define once
        :param src_num: three | two
        '''
        super(US_Encoder_parallel, self).__init__()
        if (model == 'resnet18'):
            model_resnet = resnet18(pretrained=pretrained)
            self.conv1 = model_resnet.conv1
            self.bn1 = model_resnet.bn1
            self.relu = model_resnet.relu
            self.maxpool = model_resnet.maxpool
            self.layer1 = model_resnet.layer1
            self.cwsa1 = CWSA(64)
            self.layer2 = model_resnet.layer2
            self.cwsa2 = CWSA(128)
            self.layer3 = model_resnet.layer3
            self.cwsa3 = CWSA(256)
            self.layer4 = model_resnet.layer4
            self.cwsa4 = CWSA(512)
            self.sc = StyleCross(src_num=src_num, crop='neither')
            self.aug_code = aug_code
            assert self.aug_code in ['1011','1101','1110', '1111']
        else:
            print('Wrong Name!')

    def forward(self, input, train=True, norm_flag=True):
        if train:
            '''Org'''
            fu = self.conv1(input)
            fu = self.bn1(fu)
            fu = self.relu(fu)
            fu = self.maxpool(fu)
            fu1 = self.layer1(fu)
            fu1 = self.cwsa1(fu1)
            fu2 = self.layer2(fu1)
            fu2 = self.cwsa2(fu2)
            fu3 = self.layer3(fu2)
            fu3 = self.cwsa3(fu3)
            fu4 = self.layer4(fu3)
            fu4 = self.cwsa4(fu4)

            '''Aug Low'''
            if self.aug_code in ['1011','1101','1111']:
                self.sc.active = True
                invariant_livenessL0 = self.sc(input,mode='liveness')
                invariant_livenessL0 = self.conv1(invariant_livenessL0)
                invariant_livenessL0 = self.bn1(invariant_livenessL0)
                invariant_livenessL0 = self.relu(invariant_livenessL0)
                invariant_livenessL0 = self.maxpool(invariant_livenessL0)
                self.sc.active = True
                invariant_livenessL0 = self.sc(invariant_livenessL0, mode='liveness')
                invariant_livenessL1 = self.layer1(invariant_livenessL0)
                invariant_livenessL1 = self.cwsa1(invariant_livenessL1)
                invariant_livenessL2 = self.layer2(invariant_livenessL1)
                invariant_livenessL2 = self.cwsa2(invariant_livenessL2)
                invariant_livenessL3 = self.layer3(invariant_livenessL2)
                invariant_livenessL3 = self.cwsa3(invariant_livenessL3)
                invariant_livenessL4 = self.layer4(invariant_livenessL3)
                invariant_livenessL4 = self.cwsa4(invariant_livenessL4)
            else:
                invariant_livenessL4 = torch.zeros_like(fu4)

            '''Aug Middle'''
            if self.aug_code in ['1011','1110','1111']:
                self.sc.active = True
                invariant_livenessM1 = self.sc(fu1, mode='liveness')
                invariant_livenessM2 = self.layer2(invariant_livenessM1)
                invariant_livenessM2 = self.cwsa2(invariant_livenessM2)
                self.crossnorm.active = True
                invariant_livenessM2 = self.sc(invariant_livenessM2,mode='liveness')
                invariant_livenessM3 = self.layer3(invariant_livenessM2)
                invariant_livenessM3 = self.cwsa3(invariant_livenessM3)
                invariant_livenessM4 = self.layer4(invariant_livenessM3)
                invariant_livenessM4 = self.cwsa4(invariant_livenessM4)
            else:
                invariant_livenessM4 = torch.zeros_like(fu4)

            '''Aug High'''
            if self.aug_code in ['1101','1110','1111']:
                self.sc.active = True
                invariant_livenessH3 = self.sc(fu3, mode='liveness')
                invariant_livenessH4 = self.layer4(invariant_livenessH3)
                invariant_livenessH4 = self.cwsa4(invariant_livenessH4)
                self.sc.active = True
                invariant_livenessH4 = self.sc(invariant_livenessH4,mode='liveness')
            else:
                invariant_livenessH4 = torch.zeros_like(fu4)


            fu4 = fu4.view(fu4.shape[0], -1)
            # if aug features are all zero, then they do not participate contrast
            invariant_livenessL4 = invariant_livenessL4.view(fu4.shape[0], -1)
            invariant_livenessM4 = invariant_livenessM4.view(fu4.shape[0], -1)
            invariant_livenessH4 = invariant_livenessH4.view(fu4.shape[0], -1)

            if norm_flag:
                fu4 = F.normalize(fu4, dim=-1)
                invariant_livenessL4 = F.normalize(invariant_livenessL4, dim=-1)
                invariant_livenessM4 = F.normalize(invariant_livenessM4, dim=-1)
                invariant_livenessH4 = F.normalize(invariant_livenessH4, dim=-1)

            return fu4, invariant_livenessH4, invariant_livenessM4, invariant_livenessL4
        else:
            fu = self.conv1(input)
            fu = self.bn1(fu)
            fu = self.relu(fu)
            fu = self.maxpool(fu)
            fu = self.layer1(fu)
            fu = self.cwsa1(fu)
            fu = self.layer2(fu)
            fu = self.cwsa2(fu)
            fu = self.layer3(fu)
            fu = self.cwsa3(fu)
            fu = self.layer4(fu)
            fu = self.cwsa4(fu)

            fu = fu.view(fu.shape[0], -1)
            if norm_flag:
                fu = F.normalize(fu, dim=-1)
            return fu

class US_Encoder_cascade(nn.Module):
    def __init__(self, model, pretrained, aug_code='0110', src_num='three'):
        '''
        :param model: resnet18
        :param pretrained: True | False
        :param aug_code:  The first digit of the binary code indicates mul(cascade 0) or add(parallel 1),the remaining three represent whether a Style Cross occurred at the corresponding level
        add mode { '1011': L+M,'1101':L+H ,'1110':M+H, '1111':L+M+H } mul mode { '0001': L,'0010':M ,'0100':H, '0011':L*M, '0101':L*H, '0110':M*H, '0111':L*M*H }
        '0000' '1000' -> org
        '1001' '1010' '1100' -> Repeat in add and mul modes, only define once
        :param src_num: three | two
        '''
        super(US_Encoder_cascade, self).__init__()
        if (model == 'resnet18'):
            model_resnet = resnet18(pretrained=pretrained)
            self.conv1 = model_resnet.conv1
            self.bn1 = model_resnet.bn1
            self.relu = model_resnet.relu
            self.maxpool = model_resnet.maxpool
            self.layer1 = model_resnet.layer1
            self.cwsa1 = CWSA(64)
            self.layer2 = model_resnet.layer2
            self.cwsa2 = CWSA(128)
            self.layer3 = model_resnet.layer3
            self.cwsa3 = CWSA(256)
            self.layer4 = model_resnet.layer4
            self.cwsa4 = CWSA(512)
            self.sc = StyleCross(src_num=src_num, crop='neither')
            self.aug_code = aug_code
            assert self.aug_code in ['0001','0010','0100', '0011', '0101', '0110', '0111']
        else:
            print('Wrong Name!')

    def forward(self, input, train=True, norm_flag=True):
        if train:
            '''Org Flow'''
            fu = self.conv1(input)
            fu = self.bn1(fu)
            fu = self.relu(fu)
            fu = self.maxpool(fu)
            fu1 = self.layer1(fu)
            fu1 = self.cwsa1(fu1)
            fu2 = self.layer2(fu1)
            fu2 = self.cwsa2(fu2)
            fu3 = self.layer3(fu2)
            fu3 = self.cwsa3(fu3)
            fu4 = self.layer4(fu3)
            fu4 = self.cwsa4(fu4)

            '''Aug Low'''
            if self.aug_code in ['0001','0011','0101','0111']:
                self.sc.active = True
                invariant_livenessL0 = self.sc(input,mode='liveness')
                invariant_livenessL0 = self.conv1(invariant_livenessL0)
                invariant_livenessL0 = self.bn1(invariant_livenessL0)
                invariant_livenessL0 = self.relu(invariant_livenessL0)
                invariant_livenessL0 = self.maxpool(invariant_livenessL0)
                self.sc.active = True
                invariant_livenessL0 = self.sc(invariant_livenessL0, mode='liveness')
                invariant_livenessL1 = self.layer1(invariant_livenessL0)
                invariant_livenessL1 = self.cwsa1(invariant_livenessL1)
            else:
                invariant_livenessL1 = fu1

            '''Aug Middle'''
            if self.aug_code in ['0010','0011','0110','0111']:
                self.sc.active = True
                invariant_livenessM1 = self.sc(invariant_livenessL1, mode='liveness')
                invariant_livenessM2 = self.layer2(invariant_livenessM1)
                invariant_livenessM2 = self.cwsa2(invariant_livenessM2)
                self.crossnorm.active = True
                invariant_livenessM2 = self.sc(invariant_livenessM2,mode='liveness')
                invariant_livenessM3 = self.layer3(invariant_livenessM2)
                invariant_livenessM3 = self.cwsa3(invariant_livenessM3)
            elif self.aug_code in ['0001','0101']:
                invariant_livenessM2 = self.layer2(invariant_livenessL1)
                invariant_livenessM2 = self.cwsa2(invariant_livenessM2)
                invariant_livenessM3 = self.layer3(invariant_livenessM2)
                invariant_livenessM3 = self.cwsa3(invariant_livenessM3)
            else:
                invariant_livenessM3 = fu3


            '''Aug High'''
            if self.aug_code in ['0100','0101','0110','0111']:
                self.sc.active = True
                invariant_livenessH3 = self.sc(invariant_livenessM3, mode='liveness')
                invariant_livenessH4 = self.layer4(invariant_livenessH3)
                invariant_livenessH4 = self.cwsa4(invariant_livenessH4)
                self.sc.active = True
                invariant_livenessH4 = self.sc(invariant_livenessH4,mode='liveness')
            else:
                invariant_livenessH4 = self.layer4(invariant_livenessM3)
                invariant_livenessH4 = self.cwsa4(invariant_livenessH4)


            fu4 = fu4.view(fu4.shape[0], -1)
            invariant_livenessH4 = invariant_livenessH4.view(fu4.shape[0], -1)

            if norm_flag:
                fu4 = F.normalize(fu4, dim=-1)
                invariant_livenessH4 = F.normalize(fu4, dim=-1)

            return fu4, invariant_livenessH4
        else:
            fu = self.conv1(input)
            fu = self.bn1(fu)
            fu = self.relu(fu)
            fu = self.maxpool(fu)
            fu = self.layer1(fu)
            fu = self.cwsa1(fu)
            fu = self.layer2(fu)
            fu = self.cwsa2(fu)
            fu = self.layer3(fu)
            fu = self.cwsa3(fu)
            fu = self.layer4(fu)
            fu = self.cwsa4(fu)

            fu = fu.view(fu.shape[0], -1)
            if norm_flag:
                fu = F.normalize(fu, dim=-1)
            return fu

class VS_Encoder_parallel(nn.Module):
    def __init__(self, model, pretrained, aug_code='1110', src_num='three'):
        '''
        :param model: resnet18
        :param pretrained: True | False
        :param aug_code:  The first digit of the binary code indicates mul(cascade 0) or add(parallel 1),the remaining three represent whether a Style Cross occurred at the corresponding level
        add mode { '1011': L+M,'1101':L+H ,'1110':M+H, '1111':L+M+H } mul mode { '0001': L,'0010':M ,'0100':H, '0011':L*M, '0101':L*H, '0110':M*H, '0111':L*M*H }
        '0000' '1000' -> org
        '1001' '1010' '1100' -> Repeat in add and mul modes, only define once
        :param src_num: three | two
        '''
        super(VS_Encoder_parallel, self).__init__()
        if (model == 'resnet18'):
            model_resnet = resnet18(pretrained=pretrained)
            self.conv1 = model_resnet.conv1
            self.bn1 = model_resnet.bn1
            self.relu = model_resnet.relu
            self.maxpool = model_resnet.maxpool
            self.layer1 = model_resnet.layer1
            self.layer2 = model_resnet.layer2
            self.layer3 = model_resnet.layer3
            self.layer4 = model_resnet.layer4
            self.sc = StyleCross(src_num=src_num, crop='neither')
            self.aug_code = aug_code
            assert self.aug_code in ['1011', '1101', '1110', '1111']
        else:
            print('Wrong Name!')

    def forward(self, input, train=True, norm_flag=True):
        if train:
            '''Org'''
            fv = self.conv1(input)
            fv = self.bn1(fv)
            fv = self.relu(fv)
            fv = self.maxpool(fv)
            fv1 = self.layer1(fv)
            fv2 = self.layer2(fv1)
            fv3 = self.layer3(fv2)
            fv4 = self.layer4(fv3)

            '''Aug Low'''
            if self.aug_code in ['1011', '1101', '1111']:
                self.sc.active = True
                invariant_idL0 = self.sc(input, mode='id')
                invariant_idL0 = self.conv1(invariant_idL0)
                invariant_idL0 = self.bn1(invariant_idL0)
                invariant_idL0 = self.relu(invariant_idL0)
                invariant_idL0 = self.maxpool(invariant_idL0)
                self.sc.active = True
                invariant_idL0 = self.sc(invariant_idL0, mode='id')
                invariant_idL1 = self.layer1(invariant_idL0)
                invariant_idL2 = self.layer2(invariant_idL1)
                invariant_idL3 = self.layer3(invariant_idL2)
                invariant_idL4 = self.layer4(invariant_idL3)
            else:
                invariant_idL4 = torch.zeros_like(fv4)

            '''Aug Middle'''
            if self.aug_code in ['1011', '1110', '1111']:
                self.sc.active = True
                invariant_idM1 = self.sc(fv1, mode='id')
                invariant_idM2 = self.layer2(invariant_idM1)
                self.sc.active = True
                invariant_idM2 = self.sc(invariant_idM2, mode='id')
                invariant_idM3 = self.layer3(invariant_idM2)
                invariant_idM4 = self.layer4(invariant_idM3)
            else:
                invariant_idM4 = torch.zeros_like(fv4)

            '''Aug High'''
            if self.aug_code in ['1101', '1110', '1111']:
                self.sc.active = True
                invariant_idH3 = self.sc(fv3, mode='id')
                invariant_idH4 = self.layer4(invariant_idH3)
                self.sc.active = True
                invariant_idH4 = self.sc(invariant_idH4, mode='id')
            else:
                invariant_idH4 = torch.zeros_like(fv4)

            fv4 = fv4.view(fv4.shape[0], -1)
            invariant_idL4 = invariant_idL4.view(fv4.shape[0], -1)
            invariant_idM4 = invariant_idM4.view(fv4.shape[0], -1)
            invariant_idH4 = invariant_idH4.view(fv4.shape[0], -1)
            if norm_flag:
                fv4 = F.normalize(fv4, dim=-1)
                invariant_idL4 = F.normalize(invariant_idL4, dim=-1)
                invariant_idM4 = F.normalize(invariant_idM4, dim=-1)
                invariant_idH4 = F.normalize(invariant_idH4, dim=-1)
            return fv4, invariant_idH4, invariant_idM4, invariant_idL4
        else:
            fv = self.conv1(input)
            fv = self.bn1(fv)
            fv = self.relu(fv)
            fv = self.maxpool(fv)
            fv = self.layer1(fv)
            fv = self.layer2(fv)
            fv = self.layer3(fv)
            fv = self.layer4(fv)

            fv = fv.view(fv.shape[0], -1)
            if norm_flag:
                fv = F.normalize(fv, dim=-1)
            return fv

class VS_Encoder_cascade(nn.Module):
    def __init__(self, model, pretrained, aug_code='0110', src_num='three'):
        '''
        :param model: resnet18
        :param pretrained: True | False
        :param aug_code:  The first digit of the binary code indicates mul(cascade 0) or add(parallel 1),the remaining three represent whether a Style Cross occurred at the corresponding level
        add mode { '1011': L+M,'1101':L+H ,'1110':M+H, '1111':L+M+H } mul mode { '0001': L,'0010':M ,'0100':H, '0011':L*M, '0101':L*H, '0110':M*H, '0111':L*M*H }
        '0000' '1000' -> org
        '1001' '1010' '1100' -> Repeat in add and mul modes, only define once
        :param src_num: three | two
        '''
        super(VS_Encoder_cascade, self).__init__()
        if (model == 'resnet18'):
            model_resnet = resnet18(pretrained=pretrained)
            self.conv1 = model_resnet.conv1
            self.bn1 = model_resnet.bn1
            self.relu = model_resnet.relu
            self.maxpool = model_resnet.maxpool
            self.layer1 = model_resnet.layer1
            self.layer2 = model_resnet.layer2
            self.layer3 = model_resnet.layer3
            self.layer4 = model_resnet.layer4
            self.sc = StyleCross(src_num=src_num, crop='neither')
            self.aug_code = aug_code
            assert self.aug_code in ['0001','0010','0100', '0011', '0101', '0110', '0111']
        else:
            print('Wrong Name!')

    def forward(self, input, train=True, norm_flag=True):
        if train:
            '''Org'''
            fv = self.conv1(input)
            fv = self.bn1(fv)
            fv = self.relu(fv)
            fv = self.maxpool(fv)
            fv1 = self.layer1(fv)
            fv2 = self.layer2(fv1)
            fv3 = self.layer3(fv2)
            fv4 = self.layer4(fv3)

            '''Aug Low'''
            if self.aug_code in ['0001','0011','0101','0111']:
                self.sc.active = True
                invariant_idL0 = self.sc(input, mode='id')
                invariant_idL0 = self.conv1(invariant_idL0)
                invariant_idL0 = self.bn1(invariant_idL0)
                invariant_idL0 = self.relu(invariant_idL0)
                invariant_idL0 = self.maxpool(invariant_idL0)
                self.sc.active = True
                invariant_idL0 = self.sc(invariant_idL0, mode='id')
                invariant_idL1 = self.layer1(invariant_idL0)
            else:
                invariant_idL1 = fv1

            '''Aug Middle'''
            if self.aug_code in ['0010','0011','0110','0111']:
                self.sc.active = True
                invariant_idM1 = self.sc(invariant_idL1, mode='id')
                invariant_idM2 = self.layer2(invariant_idM1)
                self.sc.active = True
                invariant_idM2 = self.sc(invariant_idM2, mode='id')
                invariant_idM3 = self.layer3(invariant_idM2)
            elif self.aug_code in ['0001','0101']:
                invariant_idM2 = self.layer2(invariant_idL1)
                invariant_idM3 = self.layer3(invariant_idM2)
            else:
                invariant_idM3 = fv3

            '''Aug High'''
            if self.aug_code in ['0100','0101','0110','0111']:
                self.sc.active = True
                invariant_idH3 = self.sc(invariant_idM3, mode='id')
                invariant_idH4 = self.layer4(invariant_idH3)
                self.sc.active = True
                invariant_idH4 = self.sc(invariant_idH4, mode='id')
            else:
                invariant_idH4 = self.layer4(invariant_idM3)

            fv4 = fv4.view(fv4.shape[0], -1)
            invariant_idH4 = invariant_idH4.view(fv4.shape[0], -1)

            if norm_flag:
                fv4 = F.normalize(fv4, dim=-1)
                invariant_idH4 = F.normalize(invariant_idH4, dim=-1)
            return fv4, invariant_idH4
        else:
            fv = self.conv1(input)
            fv = self.bn1(fv)
            fv = self.relu(fv)
            fv = self.maxpool(fv)
            fv = self.layer1(fv)
            fv = self.layer2(fv)
            fv = self.layer3(fv)
            fv = self.layer4(fv)

            fv = fv.view(fv.shape[0], -1)
            if norm_flag:
                fv = F.normalize(fv, dim=-1)
            return fv

'''Classifier & Discriminator'''
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier_layer = nn.Linear(512*8*8, 2, bias=False)
        self.classifier_layer.weight.data.normal_(0, 0.01)

    def forward(self, input, norm_flag=True):
        if norm_flag:
            for W in self.classifier_layer.parameters():
                W = F.normalize(W,dim=1)
            classifier_out = self.classifier_layer(input)
            return classifier_out
        else:
            classifier_out = self.classifier_layer(input)
            return classifier_out

class Discriminator(nn.Module):
    def __init__(self,ID_num):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Linear(512*8*8, ID_num, bias=False)
        self.discriminator.weight.data.normal_(0, 0.3)

    def forward(self, fv, norm_flag=True):
        if norm_flag:
            for W in self.discriminator.parameters():
                W = F.normalize(W,dim=1)
            dout = self.discriminator(fv)
        else:
            dout = self.discriminator(fv)
        return dout


if __name__ == '__main__':
    from thop import profile
    x = Variable(torch.ones(1, 3, 256, 256))
    block = BasicBlock(256,512,2,downsample=nn.Conv2d(256, 512, kernel_size=1, stride=2,bias=False)).cuda()
    # print(block)
    # total = sum([param.nelement() for param in block.parameters()])
    # print("Number of BasicBlock parameter: %.2fM" % (total/1e6))
    inputs = torch.randn(1,256,16,16).cuda()
    flops, params = profile(block, (inputs,))
    print('flops: %.2fG' % (flops / 1e9), 'params: %.2fM' % (params/1e6))







