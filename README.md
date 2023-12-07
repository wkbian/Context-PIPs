# Context-PIPs: Persistent Independent Particles Demands Spatial Context Features

This repository is the source code for paper [Context-PIPs: Persistent Independent Particles Demands Spatial Context Features](https://arxiv.org/abs/2306.02000), NeurIPS 2023.

[Weikang Bian](https://wkbian.github.io/)<sup>\*</sup>,
[Zhaoyang Huang](https://drinkingcoder.github.io/)<sup>\*</sup>,
[Xiaoyu Shi](https://xiaoyushi97.github.io/),
Yitong Dong,
[Yijin Li](https://eugenelyj.github.io/),
[Hongsheng Li](https://www.ee.cuhk.edu.hk/~hsli/)
 ( <sup>\*</sup> denotes equal contributions.)

<img src='https://wkbian.github.io/Projects/Context-PIPs/images/motocross-jump_context-tap.gif'>

**[[Paper](https://arxiv.org/abs/2306.02000)] [[Project Page](https://wkbian.github.io/Projects/Context-PIPs/)]**


## Requirements
```bash
conda create --name context_pips python=3.10
conda activate context_pips
conda install pytorch=1.12.0 torchvision=0.13.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## Data Preparation

To evaluate/train our Context-PIPs, you will need to download the following datasets.

* [FlyingThings++](https://github.com/aharley/pips#flyingthings-dataset)
* [CroHD](https://motchallenge.net/data/Head_Tracking_21/)
* [TAP-Vid](https://github.com/google-deepmind/tapnet#tap-vid-benchmark) (optional)

You can create symbolic links to wherever the datasets were downloaded in the `data` folder.

```text
├── data
    ├── flyingthings
        ├── frames_cleanpass_webp
        ├── object_index
        ├── occluders_al
        ├── optical_flow
        ├── trajs_ad
    ├── HT21
        ├── test
        ├── train
```

## Evaluation

We provide a [model](https://drive.google.com/file/d/1TnRXUC6UnPQ3ak7JhFE66tu66k3O3Stw/view?usp=sharing) for evaluation.

```bash
# Evaluate Context-PIPs on FlyingThings++
python test_on_flt.py --init_dir path_to_checkpoint_folder

# Evaluate Context-PIPs on CroHD
# Occluded
python test_on_crohd.py --init_dir path_to_checkpoint_folder
# Visible
python test_on_crohd.py --init_dir path_to_checkpoint_folder --req_occlusion False
```

## Training

Similiar to PIPs, we train our model on the FlyingThings++ dataset:

```bash
python train.py \
    --horz_flip=True --vert_flip=True \
    --device_ids=\[0,1,2,3,4,5,6,7\] \
    --exp_name contextpips \
    --B 4 --N 128 --I 6 --lr 3e-4
```

## Acknowledgement

In this project, we use parts of code in:

* [PIPs](https://github.com/aharley/pips)
* [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official)

Thanks to the authors for open sourcing their code.

## Citation

```text
@inproceedings{weikang2023context,
  title={Context-PIPs: Persistent Independent Particles Demands Context Features},
  author={Weikang, BIAN and Huang, Zhaoyang and Shi, Xiaoyu and Dong, Yitong and Li, Yijin and Li, Hongsheng},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

