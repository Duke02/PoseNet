# PoseNet
A recreation of PoseNet as referenced in [this research paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kendall_PoseNet_A_Convolutional_ICCV_2015_paper.pdf?utm_source=mandiner&utm_medium=link&utm_campaign=mandiner_digit_201512) and implemented [here](https://github.com/alexgkendall/caffe-posenet). This will be used to determine the distance a visual input is from an object, as well as the input's orientation. 

## Libraries Used
* [PyTorch](https://pytorch.org/) (0.4.0) 
  * (Dependencies required of this version are implicitly included)
* [Singularity](https://github.com/singularityware/singularity) (2.5.1)
  * The recipes use Ubuntu 16.04 from [Docker](https://index.docker.io/library/ubuntu/)
    * requires [CUDA (8.0) and CUDNN (7.0) installed alongside Ubuntu:16.04](https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/8.0/devel/cudnn7/Dockerfile)
* [Python](https://www.python.org/) (3.6)
* [Anaconda](https://www.anaconda.com/) (5.1)
  * This is not required but this is how we installed our packages into the Singularity container, and is recommended. 
* matplotlib (2.2)
  * Required because of plotting.
* [torchviz](https://github.com/szagoruyko/pytorchviz) (v0.0.1)
  * Not required but if you want to visualize the network, you must install this library using pip.

## Branches 
The branches are described in this section.
* **Master** - Master branch. This branch is protected from pushes from all and merges from outside sources 
to protect its integrity. If you want to make an edit to the master branch, submit an issue and request for 
someone within the project to work on it. Merge requests can only be made from Masters and no one can push to it.
* **Caffe** - The branch that uses Caffe. If it is decided to switch back to using Caffe, this branch is to prevent
such a decision from forcing project contributors to start from scratch. This branch likely doesn't work
and is protected to preserve its integrity. Caffe is protected to allow merges from only
Masters and pushes from no one.  
* **dev** - The development branch that is the precursor to master branch.

## Usage
To get all avaiable arguments, run `python3 main.py --help` to get all the arguments of the program.

To visualize the network, run `jupyter notebook` and open `viewPoseNet.ipynb`. For an already existing visualization,
open `visualize/<version>/viewPoseNet.md`.

## Pretrained model
The pretrained model is the GoogLeNet model provided by [hazirbas at GitHub](https://github.com/hazirbas/posenet-pytorch)