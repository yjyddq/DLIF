import torch
import torch.nn as nn
import torch.nn.functional as F

class OrthogonalLoss(nn.Module):
    def __init__(self):
        super(OrthogonalLoss, self).__init__()
    def forward(self, fu, fv): # [B,D] [B,D]
        innerdot = torch.matmul(fu,fv.T) # [B,B]
        loss = torch.mean(innerdot**2)
        return loss

class AMSoftmaxLoss(nn.Module):
    '''drives from the PatchNet:
    https://arxiv.org/abs/2203.14325
    '''
    def __init__(self, s=30.0, m_l=0.4, m_s=0.1):
        '''
        AM Softmax Loss
        '''
        super(AMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = [m_s, m_l]

    def forward(self, score, labels):
        '''
        input:
            score shape (N, class)
            labels shape (N)
        '''
        assert len(score) == len(labels)
        assert torch.min(labels) >= 0

        m = torch.tensor([self.m[ele] for ele in labels]).to(score.device)
        numerator = self.s * (torch.diagonal(score.transpose(0, 1)[labels]) - m)
        excl = torch.cat([torch.cat((score[i, :y], score[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)],
                         dim=0) # (N.1)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)

        L = numerator - torch.log(denominator)

        return - torch.mean(L)

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.02, contrast_mode='all',
                 base_temperature=0.02):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # 2N,C
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature) # 2N,2N

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count) # 2N,2N

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        ) # torch.scatter(input: Tensor, dim: Union[str, ellipsis, None], index: Tensor, value: Number)

        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits * mask - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


if __name__ == '__main__':
    import numpy as np
    torch.set_printoptions(threshold=np.inf)
    suploss = SupConLoss()
    x = torch.randn((5,2,512),requires_grad=True)
    x = F.normalize(x,dim=-1)
    labels = torch.randint(0,2,(5,))
    loss = suploss(x.detach(),labels)

