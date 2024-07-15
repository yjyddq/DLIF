class DefaultConfigs(object):
    protocol = 'O_C_M_to_I'
    global_ID_num = 20+20+15
    seed = 666
    # SGD
    weight_decay = 5e-4
    momentum = 0.9
    # learning rate
    init_lr = 0.0005
    final_lr = 0.00001
    step = 50
    gamma = 0.5
    # model
    pretrained = True
    model = 'resnet18'
    # training parameters
    gpus = '0,1'
    optim = 'AdamW'
    # To be noticed, we set the training batchsize of DataLoader to be constant to 1,
    # because we take multiple samples of multiple identities to be equivalent to a mini-batch
    batch_size = 1
    test_batch_size = 128
    max_iter = 15000
    iter_per_epoch = 50
    warmup_epochs = 10
    lambda_ortho = 1
    lambda_amb = 1
    lambda_aaicu = 1
    lambda_aaicv = 1
    s = 30
    # test model name
    tgt_best_model_name = 'model_best_0.16667_7.pth.tar'
    # source data information
    src1_data = r'/homes/yangjy/Dataset/OCMIv2/Oulu_NPU'
    src2_data = r'/home/yangjy/Dataset/OCMIv2/CASIA_FASD'
    src3_data = r'/home/yangjy/Dataset/OCMIv2/MSU_MFSD'
    # target data information
    tgt_data = r'/home/yangjy/Dataset/OCMIv3/Replay_Attack'
    # paths information
    checkpoint_path = './' + '_checkpoint/' + model + '/DGFANet/'
    best_model_path = './' + '_checkpoint/' + model + '/best_model/'
    logs = './logs/'

config = DefaultConfigs()
