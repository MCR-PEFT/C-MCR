[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ulip-2-towards-scalable-multimodal-pre/3d-point-cloud-classification-on-scanobjectnn)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=ulip-2-towards-scalable-multimodal-pre)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ulip-learning-unified-representation-of/3d-point-cloud-classification-on-modelnet40)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-modelnet40?p=ulip-learning-unified-representation-of)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ulip-learning-unified-representation-of/3d-point-cloud-classification-on-scanobjectnn)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=ulip-learning-unified-representation-of)

# ULIP-2: Towards Scalable Multimodal Pre-training For 3D Understanding

# ULIP: Learning a Unified Representation of Language, Images, and Point Clouds for 3D Understanding

[comment]: <> (---)

Official implementation of [ULIP-2: Towards Scalable Multimodal Pre-training For 3D Understanding](https://arxiv.org/abs/2305.08275)

Official implementation of [ULIP: Learning a Unified Representation of Language, Images, and Point Clouds for 3D Understanding](https://arxiv.org/abs/2212.05171)

[Project Website](https://tycho-xue.github.io/ULIP/)

# News
[06/09/2023] "PointBERT ULIP-2 pretrained model released, please find it in the [here](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/pointbert_ULIP-2.pt)".

[06/09/2023] A smaller version of "ULIP - ShapeNet Triplets" are released at [here](https://console.cloud.google.com/storage/browser/sfr-ulip-code-release-research/shapenet-55?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false), it's around 420GB now. Check this image folder "only_rgb_depth_images", you can choose to download this subset of rendered images, which are the exact images leveraged by ULIP instead of downloading the full "rendered_images" folder (more than 1TB).

[05/22/2023] "ULIP - Objaverse Triplets" and "ULIP - ShapeNet Triplets" have been uploaded [here](https://console.cloud.google.com/storage/browser/sfr-ulip-code-release-research).

[05/14/2023] ULIP-2 has been released!

[02/28/2023] ULIP has been accepted by CVPR 2023! 🔥🔥🔥

# Animation
![Pipeline Animation](assets/pipeline_8s_timing.gif)

[comment]: <> (---)

# What is ULIP
ULIP is a Model-agnostic Multimodal Pre-training Framework, which can leverage information from other modalities (Images, Language) to improve the ability to understand 3D data without introducing any extra latency.

[comment]: <> (---)

# Pipeline
![Overall Pipeline](assets/figure2_resize.gif)

[comment]: <> (---)

# Instructions
ULIP is a highly extensible multimodal pre-training framework, and it's model-architecture agnostic, meaning you can easily plug in any 3D backbone models and pre-train it using our framework to get a jump-start for various downstreaming tasks!
## [Install environments]
We pre-train ULIP on 8 Nvidia A100 GPUs, the code is tested with CUDA==11.0 and pytorch==1.10.1\
```conda create -n ulip python=3.7.15``` \
```conda activate ulip``` \
```conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge``` \
```pip install -r requirements.txt```\
\
[optional] \
If you want to pre-train PointNeXt, we embed a modified PointNeXt codebase inside the ./models/pointnext, please do the following to install it:
```
cd ./models/pointnext/PointNeXt \
bash update.sh \
bash install.sh \
```
## [Download datasets and initialize models, put them in the right paths.]
Download the used datasets and initialize models from [here](https://console.cloud.google.com/storage/browser/sfr-ulip-code-release-research). For now, you ONLY need to download "initialize_models", "modelnet40_normal_resampled", and "shapenet-55". You might need a gmail account to access it.\
After you download the datasets and initialize models, you can choose one of the following options: \
(1) Put it in or do a soft link to the data folder, by default the data folder should have the following structure:
```
./data |
-- ModelNet40.yaml |
-- ShapeNet-55.yaml |
-- dataset_3d.py |
-- dataset_catalog.json |
-- initialize_models |
-- labels.json |
-- modelnet40_normal_resampled |
-- shapenet-55 |
-- templates.json
```
(2) Change the paths accordingly (optional to do if you don't want to put/link downloaded files in the data folder):
```
# Change the "DATA_PATH", "PC_PATH", "IMAGE_PATH"
./data/ShapeNet-55.yaml
# Change the "DATA_PATH"
./data/ModelNet40.yaml
# Change the initialize_models address
./models/ULIP_models.py
Modify this line "pretrain_slip_model = torch.load('./data/initialize_models/slip_base_100ep.pt', map_location=torch.device('cpu'))"
```


## [Pre-train 3D backbones]
**Our framework is model architecture agonistic, currently four 3D backbones are supported:** \
**Pointnet2(ssg)**\
**PointBERT**\
**PointMLP**\
**PointNeXt**\
\
Please change the script to accommodate your system accordingly, this script is used to pre-train on 8 gpus by default. You can also modify the desired output folder in the script.
```
# the scripts are named by its correspoinding 3D backbone name.
bash ./scripts/(choose your pre-train script)
```

## [Test pre-trained models for zero-shot classification on ModelNet40]
You may also change the output path in the scripts as well.

```
bash ./scripts/(choose your test script) /path/to/your/checkpoint.pt
```
You may also change the output path in the scripts as well.

## [Pre-train & Test using different number of points]
Change the npoints argument in the scripts, by default its 8192. \
**Note: Currently we use FPS to subsample the 8192 points, which might slow down the training speed. If you'd like, you can choose to cache or save the pre-processed datasets with different number of points to speed up your pre-training.**

## [Pre-train your customized 3D backbones]
There are only two things you need to change to pre-train your own customized 3D backbones: \
(1) Define your own 3D backbone in ./models folder.\
We put a template "customized_backbone" here, you can refer to the comments to see the expected input and output shapes. You can also refer to how pointnet2 is defined here. \
(2) Use or modify this "ULIP_CUSTOMIZED" class in ./models/ULIP_models.py.\
Please refer to the comments in "ULIP_CUSTOMIZED" class, it should be straightforward to follow, and please be sure to change the "pc_feat_dims" accordingly (since we are agnostic to the point cloud output feature dimensions of your customized 3D backbones).


# Pre-trained models for zero-shot classification
Zero-shot classification on ModelNet40, 8k points pre-train, 8k points test, best checkpoint:

| model                                                                                                                                                                   | top1 | top5 |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|------|
| [Pointnet2(ssg)](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/checkpoint_pointnet2_ssg.pt?authuser=0) | 57.7 | 78.9 |
| [PointMLP](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/checkpoint_pointmlp.pt?authuser=0)            | 60.0 | 79.4 |
| [PointBERT](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/checkpoint_pointbert.pt?authuser=0)          | 60.3 | 84.0 |
| [PointNeXt](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/checkpoint_pointnext.pt?authuser=0)          | 56.2 | 77.0 |
| [PointBERT_ULIP-2(xyz input)](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/pointbert_ULIP-2.pt) | 75.6 | 93.7 |
# TODO
More supported backbones will be released soon.

# License and term of use for the released pre-train datasets
The code is under https://github.com/salesforce/ULIP/blob/main/LICENSE.txt.

The released "ULIP - Objaverse Triplets" is under https://opendatacommons.org/licenses/by/1-0/, consistent with Objaverse's license.

The released "ULIP - ShapeNet Triplets" is under the terms of use from https://shapenet.org/terms, consistent with ShapeNet's terms of use.

# Citation

    @article{xue2022ulip,
      title={ULIP: Learning Unified Representation of Language, Image and Point Cloud for 3D Understanding},
      author={Xue, Le and Gao, Mingfei and Xing, Chen and Mart{\'\i}n-Mart{\'\i}n, Roberto and Wu, Jiajun and Xiong, Caiming and Xu, Ran and Niebles, Juan Carlos and Savarese, Silvio},
      journal={arXiv preprint arXiv:2212.05171},
      year={2022}
    }
    @misc{xue2023ulip2,
      title={ULIP-2: Towards Scalable Multimodal Pre-training for 3D Understanding}, 
      author={Le Xue and Ning Yu and Shu Zhang and Junnan Li and Roberto Martín-Martín and Jiajun Wu and Caiming Xiong and Ran Xu and Juan Carlos Niebles and Silvio Savarese},
      year={2023},
      eprint={2305.08275},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

# Contact
If you have any question about this project, please contact [lxue@salesforce.com](lxue@salesforce.com)
