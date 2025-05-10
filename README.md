# DLIF
The implementation of [**Generalized Face Anti-spoofing via Finer Domain Partition and Disentangling Liveness-irrelevant Factors (DLIF)**](https://arxiv.org/abs/2407.08243).

The motivation of the proposed DLIF method:
<div align=center>
<img src="https://github.com/yjyddq/DLIF/blob/main/assets/Motivation.png" width="512" height="224" />
</div>

An overview of the proposed DLIF architecture:

<div align=center>
<img src="https://github.com/yjyddq/DLIF/blob/main/assets/architecture.png" width="892" height="384" />
</div>

## Congifuration Environment
- python 3.10 
- torch 1.12.1 
- torchvision 0.13.1
- cuda 11.4

## Data

**Dataset.** 

Download the OULU-NPU, CASIA-FASD, Idiap Replay-Attack, MSU-MFSD and CASIA-Spoof datasets.

**Data Pre-processing.** 

[MTCNN algotithm](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection) is utilized for face detection and face alignment. All the detected faces are normlaize to 256x256x3, where only RGB channels are utilized for training. 


## Training

Move to the folder `./protocol/I_C_M_to_O/` and just run like this:
```python
python -m torch.distributed.launch --nproc_per_node=ngpus train.py
```

The file `config.py` contains all the hype-parameters used during training.

## Testing

Run like this:
```python
python test.py
```

## Citation
Please cite our paper if the code is helpful to your research.
```
@article{yang2024generalized,
  title={Generalized Face Anti-spoofing via Finer Domain Partition and Disentangling Liveness-irrelevant Factors},
  author={Yang, Jingyi and Yu, Zitong and Ni, Xiuming and He, Jia and Li, Hui},
  journal={arXiv preprint arXiv:2407.08243},
  year={2024}
}

@incollection{yang2024generalized,
  title={Generalized Face Anti-spoofing via Finer Domain Partition and Disentangling Liveness-irrelevant Factors},
  author={Yang, Jingyi and Yu, Zitong and Ni, Xiuming and He, Jia and Li, Hui},
  booktitle={ECAI 2024},
  pages={274--281},
  year={2024},
  publisher={IOS Press}
}
```
