# Rotation-Invariant Local-to-Global Representation Learning for 3D Point Cloud

Seohyun Kim, Jaeyou Park, Bohyung Han

Advances in Neural Information Processing Systems (NeurIPS) 2020. 


## Introduction
Our paper proposes a local-to-global representation learning algorithm for 3D point cloud data, which is appropriate to handle various geometric transformation, especially rotation. For more details, please refer to our [project page](https://cv.snu.ac.kr/research/rotation_invariant_l2g/).

```
@InProceedings{,  
author = {Kim, Seohyun, Park, Jaeyoo, and Han, Bohyoung},  
title = {Rotation-Invariant Local-to-Global Representation Learning for 3D Point Cloud},  
booktitle = {Advances in Neural Information Processing Systems},  
year = {2020}  
}  
```

## Usage
First, please download the ModelNet40 dataset. 
Install [Tensorflow](https://www.tensorflow.org/install/). Our code is based on TF1.4 GPU and python 3.6. Please refer to github page in [PointNet++](https://github.com/charlesq34/pointnet2) to complie customized TF Operators under `tf_ops` and the other requirements. 

To train a network for ModelNet40 classification run the following script:
```
CUDA_VISIBLE_DEVICES=0 python train.py  
```


After training, run the following script for model evaluation: 
```
CUDA_VISIBLE_DEVICES=0 evaluate.py --load_dir $model_dir
```


## License 
This repository is released under MIT License (see LICENSE file for details). 
