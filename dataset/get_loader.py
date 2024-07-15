import os
import torch
from torch.utils.data import DataLoader,DistributedSampler
from dataset.dataset import FASDataset_val,FASDataset_train
from utils.utils import sample_frames_val,sample_frames_train

'''Specify the number of identities for each iteration and the number of samples for each identity'''
ID_NUM = 2
BATCHSIZE_PER_ID = 3

def get_dataset(protocol, src1_data, src2_data, src3_data, tgt_data, batch_size, test_batch_size):
    '''
    protocol: LOO (leave one out)
    '''
    print('Load Source Data')
    print('Source Data: ', src1_data)
    src1_train_data = sample_frames_train(protocol,dataset_name=src1_data)
    print('Source Data: ', src2_data)
    src2_train_data = sample_frames_train(protocol,dataset_name=src2_data)
    print('Source Data: ', src3_data)
    src3_train_data = sample_frames_train(protocol,dataset_name=src3_data)

    print('Load Target Data')
    print('Target Data: ', tgt_data)
    tgt_test_data = sample_frames_val(dataset_name=tgt_data)

    # We set the training batchsize to be constant to 1, because we take multiple samples of multiple identities to be equivalent to a mini-batch
    src1_train_sampler = DistributedSampler(FASDataset_train(src1_train_data, id_num=ID_NUM, batchsize_per_id=BATCHSIZE_PER_ID), shuffle=True)
    src1_train_dataloader = DataLoader(FASDataset_train(src1_train_data, id_num=ID_NUM, batchsize_per_id=BATCHSIZE_PER_ID), batch_size=batch_size,
                                             num_workers=8,prefetch_factor=8,
                                             sampler=src1_train_sampler)


    src2_train_sampler = DistributedSampler(FASDataset_train(src2_train_data, id_num=ID_NUM, batchsize_per_id=BATCHSIZE_PER_ID), shuffle=True)
    src2_train_dataloader = DataLoader(FASDataset_train(src2_train_data, id_num=ID_NUM, batchsize_per_id=BATCHSIZE_PER_ID), batch_size=batch_size,
                                            num_workers=8,prefetch_factor=8,
                                            sampler=src2_train_sampler)


    src3_train_sampler = DistributedSampler(FASDataset_train(src3_train_data, id_num=ID_NUM, batchsize_per_id=BATCHSIZE_PER_ID), shuffle=True)
    src3_train_dataloader = DataLoader(FASDataset_train(src3_train_data, id_num=ID_NUM, batchsize_per_id=BATCHSIZE_PER_ID), batch_size=batch_size,
                                            num_workers=8,prefetch_factor=8,
                                            sampler=src3_train_sampler)


    tgt_dataloader = DataLoader(FASDataset_val(tgt_test_data), batch_size=test_batch_size,
                                num_workers=8,prefetch_factor=8,
                                shuffle=True)
    return src1_train_dataloader, src2_train_dataloader, src3_train_dataloader, tgt_dataloader

'''Specify the number of identities for each iteration and the number of samples for each identity'''
ID_NUM_limit = 3
BATCHSIZE_PER_ID_limit = 3

def get_dataset_limit_data(protocol, src1_data, src2_data, tgt_data, batch_size, test_batch_size):
    '''
    protocol: limit source data
    '''
    print('Load Source Data')
    print('Source Data: ', src1_data)
    src1_train_data = sample_frames_train(protocol,dataset_name=src1_data)
    print('Source Data: ', src2_data)
    src2_train_data = sample_frames_train(protocol,dataset_name=src2_data)

    print('Load Target Data')
    print('Target Data: ', tgt_data)
    tgt_test_data = sample_frames_val(dataset_name=tgt_data)

    # We set the training batchsize to be constant to 1, because we take multiple samples of multiple identities to be equivalent to a mini-batch
    src1_train_sampler = DistributedSampler(FASDataset_train(src1_train_data, id_num=ID_NUM_limit, batchsize_per_id=BATCHSIZE_PER_ID_limit), shuffle=True)
    src1_train_dataloader = DataLoader(FASDataset_train(src1_train_data, id_num=ID_NUM_limit, batchsize_per_id=BATCHSIZE_PER_ID_limit), batch_size=batch_size,
                                             num_workers=8,prefetch_factor=8,
                                             sampler=src1_train_sampler)


    src2_train_sampler = DistributedSampler(FASDataset_train(src2_train_data,id_num=ID_NUM_limit, batchsize_per_id=BATCHSIZE_PER_ID_limit), shuffle=True)
    src2_train_dataloader = DataLoader(FASDataset_train(src2_train_data,id_num=ID_NUM_limit, batchsize_per_id=BATCHSIZE_PER_ID_limit), batch_size=batch_size,
                                            num_workers=8,prefetch_factor=8,
                                            sampler=src2_train_sampler)


    tgt_dataloader = DataLoader(FASDataset_val(tgt_test_data), batch_size=test_batch_size,
                                num_workers=8,prefetch_factor=8,
                                shuffle=True)
    return src1_train_dataloader, src2_train_dataloader, tgt_dataloader

if __name__ == '__main__':
    import torch.distributed as dist
    from utils import sample_frames_train

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    src1_data = r'/home/yangjy/Dataset/OCMI/CASIA_FASD'
    src2_data = r'/home/yangjy/Dataset/OCMI/Oulu_NPU'
    src3_data = r'/home/yangjy/Dataset/OCMI/MSU_MFSD'
    tgt_data = r'/home/yangjy/Dataset/OCMI/Replay_Attack'







