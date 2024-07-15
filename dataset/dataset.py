import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


'''Dataset Loader'''
# Note that our data storage directory structure is as follows:
#  -CASIA-FASD\
#     -C_train\ the name of img for example (C_1_real_1_1.jpg) -> (domain_id_liveness_videoid_frameid)
#     -C_val\
#     -label.csv
#  -MSU-MFSD\
#     -M_train\
#     -M_val\
#     -label.csv\
#  -Replay-Attack\
#     -I_train\
#     -I_val\
#     -label.csv\
#  -Oulu-NPU\
#     -O_train\
#     -O_val\
#     -label.csv\
# If you find this method of loading data too cumbersome, you can modify code according to your data directory structure

class FASDataset_val(Dataset):
    def __init__(self, data_pd, transforms=None):
        self.photo_path = data_pd['photo_path'].tolist()
        self.photo_label = data_pd['photo_label'].tolist()
        self.photo_belong_to_video_ID = data_pd['photo_belong_to_video_ID'].tolist()
        self.subject_id = data_pd['subject_id'].tolist()
        if transforms is None:
            self.transforms = T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):
        cropped_img_path = self.photo_path[item]
        label = self.photo_label[item]
        videoID = self.photo_belong_to_video_ID[item]
        img = Image.open(cropped_img_path)
        img = self.transforms(img)
        return img, label, videoID

class FASDataset_train(Dataset):
    def __init__(self, video_dict, id_num, batchsize_per_id, transforms=None):
        self.video_dict = video_dict
        self.id_num = id_num
        self.batchsize_per_id = batchsize_per_id
        self.cnt = 0
        if transforms is None:
            self.transforms = T.Compose([
                T.RandomResizedCrop(256, scale=(0.2, 1.0)), # scale, rotate angle, flip prob is adjustable
                T.RandomRotation(20),
                T.RandomHorizontalFlip(0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms

    def __len__(self):
        return sum([len(self.video_dict['real'][i].keys()) for i in [k for k,v in self.video_dict['real'].items()]] +\
               [len(self.video_dict['attack'][i].keys()) for i in [k for k,v in self.video_dict['attack'].items()]])


    def __getitem__(self, item):
        # Extract id_num different IDs each time, with batchsize_per_id images for each ID,
        # which is equivalent to extracting id_num*batchsize_per_id images at once
        # The process is a litter cumbersome, you can change to your way for convenience
        id_list = np.array(list(self.video_dict['real'].keys()))
        rand_ids_index = np.random.choice(len(id_list),self.id_num,replace=False)
        id_list = id_list[rand_ids_index]

        for i,id in enumerate(id_list):
            for j in range(self.batchsize_per_id):
                rimg_list = list(self.video_dict['real'][id].keys())
                rand_rimg_index = np.random.randint(0, len(rimg_list))
                frame_id = rimg_list[rand_rimg_index]
                cropped_img_path = self.video_dict['real'][id][frame_id]['img_path']
                label = torch.tensor(self.video_dict['real'][id][frame_id]['label']).unsqueeze(0)
                tmp_id = torch.tensor(self.video_dict['real'][id][frame_id]['global_id']).unsqueeze(0)
                img = Image.open(cropped_img_path)
                img = self.transforms(img).unsqueeze(0)
                if j == 0:
                    rimgs = img
                    rlabels = label
                    rids = tmp_id
                else:
                    rimgs = torch.cat([rimgs, img], dim=0)
                    rlabels = torch.cat([rlabels, label], dim=0)
                    rids = torch.cat([rids, tmp_id], dim=0)

                aimg_list = list(self.video_dict['attack'][id].keys())
                rand_aimg_index = np.random.randint(0, len(aimg_list))
                frame_id = aimg_list[rand_aimg_index]
                cropped_img_path = self.video_dict['attack'][id][frame_id]['img_path']
                label = torch.tensor(self.video_dict['attack'][id][frame_id]['label']).unsqueeze(0)
                tmp_id = torch.tensor(self.video_dict['attack'][id][frame_id]['global_id']).unsqueeze(0)
                img = Image.open(cropped_img_path)
                img = self.transforms(img).unsqueeze(0)
                if j == 0:
                    aimgs = img
                    alabels = label
                    aids = tmp_id
                else:
                    aimgs = torch.cat([aimgs, img], dim=0)
                    alabels = torch.cat([alabels, label], dim=0)
                    aids = torch.cat([aids, tmp_id], dim=0)
            if i == 0:
                real_imgs = rimgs
                real_labels = rlabels
                real_ids = rids
                attack_imgs = aimgs
                attack_labels = alabels
                attack_ids = aids
            else:
                real_imgs = torch.cat([real_imgs, rimgs], dim=0)
                real_labels = torch.cat([real_labels, rlabels], dim=0)
                real_ids = torch.cat([real_ids, rids], dim=0)
                attack_imgs = torch.cat([attack_imgs, aimgs], dim=0)
                attack_labels = torch.cat([attack_labels, alabels], dim=0)
                attack_ids = torch.cat([attack_ids, aids], dim=0)

        return real_imgs, real_labels, real_ids, attack_imgs, attack_labels, attack_ids


if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from utils.utils import sample_frames_val,sample_frames_train
    from torch.utils.data import DataLoader
    src = r'/home/yangjy/Dataset/OCMI/CASIA_FASD'
    protocol = 'O_C_M_to_I'
    id_num = 2
    batch_size = 2
    video_dict = sample_frames_train(protocol,src)
    print(video_dict['real'].keys())
    print(video_dict['attack'].keys())
    dataset = DataLoader(FASDataset_train(video_dict, id_num, batch_size),batch_size=1,shuffle=True)
    print(len(dataset))
    for i,(rimgs, rlabels, rids, aimgs, alabels, aids) in enumerate(dataset):
        print(rimgs.shape)
        print(aids)
        print(rids)
        break



