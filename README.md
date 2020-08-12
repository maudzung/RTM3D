# RTM3D-PyTorch

[![python-image]][python-url]
[![pytorch-image]][pytorch-url]

The PyTorch Implementation of the paper: 
[RTM3D: Real-time Monocular 3D Detection from Object Keypoints for Autonomous Driving](https://arxiv.org/pdf/2001.03343.pdf) (ECCV 2020)

---

## Demonstration

![demo](./docs/demo.gif)

## Features
- [x] Realtime 3D object detection based on a monocular RGB image
- [x] Support [distributed data parallel training](https://github.com/pytorch/examples/tree/master/distributed/ddp)
- [x] Tensorboard
- [x] ResNet-based **K**eypoint **F**eature **P**yramid **N**etwork (KFPN) (Using by setting `--arch fpn_resnet_18`)
- [ ] Use images from both left and right cameras (Control by setting the `use_left_cam_prob` argument)
- [ ] Release pre-trained models 



## Some modifications from the paper
- _**Formula (3)**_:  
   - A negative value can't be an input of the `log` operator, so please **don't normalize dim** as mentioned in
the paper because the normalized dim values maybe less than `0`. Hence I've directly regressed to absolute dimension values in meters.
   - Use `L1 loss` for depth estimation (applying the `sigmoid` activation to the depth output first).

- _**Formula (5)**_: I haven't taken the absolute values of the ground-truth, 
I have used the **relative values** instead. [The code is here](https://github.com/maudzung/RTM3D/blob/45b9d8af1298a6ad7dacb99a8f538f285696ded4/src/data_process/kitti_dataset.py#L284)

- _**Formula (7)**_: `argmin` instead of `argmax`

- Generate heatmap for the center and vertexes of objects as the CenterNet paper. If you want to use the strategy from RTM3D paper,
you can pass the `dynamic-sigma` argument to the `train.py` script.


## 2. Getting Started
### 2.1. Requirement

```shell script
pip install -U -r requirements.txt
```

### 2.2. Data Preparation
Download the 3D KITTI detection dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

The downloaded data includes:

- Training labels of object data set _**(5 MB)**_
- Camera calibration matrices of object data set _**(16 MB)**_
- **Left color images** of object data set _**(12 GB)**_
- **Right color images** of object data set _**(12 GB)**_

Please make sure that you construct the source code & dataset directories structure as below.

### 2.3. RTM3D architecture


![architecture](./docs/rtm3d_architecture.png)


The model takes **only the RGB images** as the input and outputs the `main center heatmap`, `vertexes heatmap`, 
and `vertexes coordinate` as the base module to estimate `3D bounding box`.

### 2.4. How to run

#### 2.4.1. Visualize the dataset 

```shell script
cd src/data_process
```

- To visualize camera images with 3D boxes, let's execute:

```shell script
python kitti_dataset.py
```

Then _Press **n** to see the next sample >>> Press **Esc** to quit..._


#### 2.4.2. Inference

Download the trained model from [**_here_**](https://drive.google.com/drive/folders/1lKOLHhWZasoC7cKNLcB714LBDS91whCr?usp=sharing) (will be released),
then put it to `${ROOT}/checkpoints/` and execute:

```shell script
python test.py --gpu_idx 0 --arch resnet_18 --pretrained_path ../checkpoints/rtm3d_resnet_18.pth
```

#### 2.4.3. Evaluation

```shell script
python evaluate.py --gpu_idx 0 --arch resnet_18 --pretrained_path <PATH>
```

#### 2.4.4. Training

##### 2.4.4.1. Single machine, single gpu

```shell script
python train.py --gpu_idx 0 --arch <ARCH> --batch_size <N> --num_workers <N>...
```

##### 2.4.4.2. Multi-processing Distributed Data Parallel Training
We should always use the `nccl` backend for multi-processing distributed training since it currently provides the best 
distributed training performance.

- **Single machine (node), multiple GPUs**

```shell script
python train.py --dist-url 'tcp://127.0.0.1:29500' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0
```

- **Two machines (two nodes), multiple GPUs**

_**First machine**_

```shell script
python train.py --dist-url 'tcp://IP_OF_NODE1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 0
```
_**Second machine**_

```shell script
python train.py --dist-url 'tcp://IP_OF_NODE2:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 1
```

To reproduce the results, you can run the bash shell script

```bash
./train.sh
```


#### Tensorboard

- To track the training progress, go to the `logs/` folder and 

```shell script
cd logs/<saved_fn>/tensorboard/
tensorboard --logdir=./
```

- Then go to [http://localhost:6006/](http://localhost:6006/):


## Contact

If you think this work is useful, please give me a star! <br>
If you find any errors or have any suggestions, please contact me (**Email:** `nguyenmaudung93.kstn@gmail.com`). <br>
Thank you!


## Citation

```bash
@article{RTM3D,
  author = {Peixuan Li,  Huaici Zhao, Pengfei Liu, Feidao Cao},
  title = {RTM3D: Real-time Monocular 3D Detection from Object Keypoints for Autonomous Driving},
  year = {2020},
  conference = {ECCV 2020},
}
@misc{RTM3D-PyTorch,
  author =       {Nguyen Mau Dung},
  title =        {{RTM3D-PyTorch: PyTorch Implementation of the RTM3D paper}},
  howpublished = {\url{https://github.com/maudzung/RTM3D-PyTorch}},
  year =         {2020}
}
```

## References

[1] CenterNet: [Objects as Points paper](https://arxiv.org/abs/1904.07850), [PyTorch Implementation](https://github.com/xingyizhou/CenterNet)

## Folder structure

```
${ROOT}
└── checkpoints/    
    ├── rtm3d_resnet_18.pth
    ├── rtm3d_fpn_resnet_18.pth
└── dataset/    
    └── kitti/
        ├──ImageSets/
        │   ├── test.txt
        │   ├── train.txt
        │   └── val.txt
        ├── training/
        │   ├── image_2/ (left color camera)
        │   ├── image_3/ (right color camera)
        │   ├── calib/
        │   ├── label_2/
        └── testing/  
        │   ├── image_2/ (left color camera)
        │   ├── image_3/ (right color camera)
        │   ├── calib/
        └── classes_names.txt
└── src/
    ├── config/
    │   ├── train_config.py
    │   └── kitti_config.py
    ├── data_process/
    │   ├── kitti_dataloader.py
    │   ├── kitti_dataset.py
    │   └── kitti_data_utils.py
    ├── models/
    │   ├── fpn_resnet.py
    │   ├── resnet.py
    │   ├── model_utils.py
    └── utils/
    │   ├── evaluation_utils.py
    │   ├── logger.py
    │   ├── misc.py
    │   ├── torch_utils.py
    │   ├── train_utils.py
    ├── evaluate.py
    ├── test.py
    ├── train.py
    └── train.sh
├── README.md 
└── requirements.txt
```


## Usage

```
usage: train.py [-h] [--seed SEED] [--saved_fn FN] [--root-dir PATH]
                [--arch ARCH] [--pretrained_path PATH] [--head_conv HEAD_CONV]
                [--hflip_prob HFLIP_PROB]
                [--use_left_cam_prob USE_LEFT_CAM_PROB] [--dynamic-sigma]
                [--no-val] [--num_samples NUM_SAMPLES]
                [--num_workers NUM_WORKERS] [--batch_size BATCH_SIZE]
                [--print_freq N] [--tensorboard_freq N] [--checkpoint_freq N]
                [--start_epoch N] [--num_epochs N] [--lr_type LR_TYPE]
                [--lr LR] [--minimum_lr MIN_LR] [--momentum M] [-wd WD]
                [--optimizer_type OPTIMIZER] [--steps [STEPS [STEPS ...]]]
                [--world-size N] [--rank N] [--dist-url DIST_URL]
                [--dist-backend DIST_BACKEND] [--gpu_idx GPU_IDX] [--no_cuda]
                [--multiprocessing-distributed] [--evaluate]
                [--resume_path PATH] [--K K]

The Implementation of RTM3D using PyTorch

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           re-produce the results with seed random
  --saved_fn FN         The name using for saving logs, models,...
  --root-dir PATH       The ROOT working directory
  --arch ARCH           The name of the model architecture
  --pretrained_path PATH
                        the path of the pretrained checkpoint
  --head_conv HEAD_CONV
                        conv layer channels for output head0 for no conv
                        layer-1 for default setting: 64 for resnets and 256
                        for dla.
  --hflip_prob HFLIP_PROB
                        The probability of horizontal flip
  --use_left_cam_prob USE_LEFT_CAM_PROB
                        The probability of using the left camera
  --dynamic-sigma       If true, compute sigma based on Amax, Amin then
                        generate heamapIf false, compute radius as CenterNet
                        did
  --no-val              If true, dont evaluate the model on the val set
  --num_samples NUM_SAMPLES
                        Take a subset of the dataset to run and debug
  --num_workers NUM_WORKERS
                        Number of threads for loading data
  --batch_size BATCH_SIZE
                        mini-batch size (default: 16), this is the totalbatch
                        size of all GPUs on the current node when usingData
                        Parallel or Distributed Data Parallel
  --print_freq N        print frequency (default: 50)
  --tensorboard_freq N  frequency of saving tensorboard (default: 50)
  --checkpoint_freq N   frequency of saving checkpoints (default: 5)
  --start_epoch N       the starting epoch
  --num_epochs N        number of total epochs to run
  --lr_type LR_TYPE     the type of learning rate scheduler (cosin or
                        multi_step)
  --lr LR               initial learning rate
  --minimum_lr MIN_LR   minimum learning rate during training
  --momentum M          momentum
  -wd WD, --weight_decay WD
                        weight decay (default: 1e-6)
  --optimizer_type OPTIMIZER
                        the type of optimizer, it can be sgd or adam
  --steps [STEPS [STEPS ...]]
                        number of burn in step
  --world-size N        number of nodes for distributed training
  --rank N              node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --gpu_idx GPU_IDX     GPU index to use.
  --no_cuda             If true, cuda is not used.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N
                        processes per node, which has N GPUs. This is the
                        fastest way to use PyTorch for either single node or
                        multi node data parallel training
  --evaluate            only evaluate the model, not training
  --resume_path PATH    the path of the resumed checkpoint
  --K K                 the number of top K
```



[python-image]: https://img.shields.io/badge/Python-3.6-ff69b4.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.5-2BAF2B.svg
[pytorch-url]: https://pytorch.org/