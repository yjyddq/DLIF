# DLIF
The implementation of [**Generalized Face Anti-spoofing via Finer Domain Partition and Disentangling Liveness-irrelevant Factors (DLIF)**](https://arxiv.org/abs/2407.08243).

The motivation of the proposed DLIF method:
<div align=center>
<img src="https://github.com/yjyddq/DLIF/assets/Motivation.pdf" width="400" height="296" />
</div>

An overview of the proposed DLIF architecture:

<div align=center>
<img src="https://github.com/yjyddq/DLIF/assets/architecture.pdf" width="700" height="345" />
</div>

## Congifuration Environment
- python 3.6 
- pytorch 0.4 
- torchvision 0.2
- cuda 8.0

## Data

**Dataset.** 

Download the OULU-NPU, CASIA-FASD, Idiap Replay-Attack, MSU-MFSD and CASIA-Spoof datasets.

**Data Pre-processing.** 

[MTCNN algotithm](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection) is utilized for face detection and face alignment. All the detected faces are normlaize to 256$\times$256$\times$3, where only RGB channels are utilized for training. 


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
@misc{yang2024generalizedfaceantispoofingfiner,
      title={Generalized Face Anti-spoofing via Finer Domain Partition and Disentangling Liveness-irrelevant Factors}, 
      author={Jingyi Yang and Zitong Yu and Xiuming Ni and Jia He and Hui Li},
      year={2024},
      eprint={2407.08243},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.08243}, 
}
```




=======
# DLIF
The implementation of [**Generalized Face Anti-spoofing via Finer Domain Partition and Disentangling Liveness-irrelevant Factors (DLIF)**](https://arxiv.org/abs/2407.08243).

The motivation of the proposed DLIF method:
<div align=center>
<img src="https://github.com/yjyddq/DLIF/assets/Motivation.pdf" width="400" height="296" />
</div>

An overview of the proposed DLIF architecture:

<div align=center>
<img src="https://github.com/yjyddq/DLIF/assets/architecture.pdf" width="700" height="345" />
</div>

## Congifuration Environment
- python 3.6 
- pytorch 0.4 
- torchvision 0.2
- cuda 8.0

## Data

**Dataset.** 

Download the OULU-NPU, CASIA-FASD, Idiap Replay-Attack, MSU-MFSD and CASIA-Spoof datasets.

**Data Pre-processing.** 

[MTCNN algotithm](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection) is utilized for face detection and face alignment. All the detected faces are normlaize to 256$\times$256$\times$3, where only RGB channels are utilized for training. 


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
@misc{yang2024generalizedfaceantispoofingfiner,
      title={Generalized Face Anti-spoofing via Finer Domain Partition and Disentangling Liveness-irrelevant Factors}, 
      author={Jingyi Yang and Zitong Yu and Xiuming Ni and Jia He and Hui Li},
      year={2024},
      eprint={2407.08243},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.08243}, 
}
```




>>>>>>> c2db40b (Initial commit)
