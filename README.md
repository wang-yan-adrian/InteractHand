# InteractHand: Robust 3D Hand Mesh Reconstruction via Interaction-Aware Segmentation and Refinement
* Official Implementation of our InteractHand.

# Introduction
Below is the overall pipeline of InteractHand.
![overall pipeline](./asset/model.jpg)

# Getting Started

## Prerequisites

```
$ conda create -n InteractHand python=3.8
$ pip install numpy==1.17.4 torch==1.9.1 torchvision==0.10.1 einops chumpy opencv-python pycocotools pyrender tqdm
```
## Preparing Datasets

* Download `MANO_RIGHT.pkl` from [here](https://mano.is.tue.mpg.de/) and place at `common/utils/manopth/mano/models`.
* Download HO3D(version 2) data and annotation files [[data](https://www.tugraz.at/institute/icg/research/team-lepetit/research-projects/hand-object-3d-pose-annotation/)][[annotation files](https://drive.google.com/drive/folders/1pmRpgv38PXvlLOODtoxpTYnIpYTkNV6b?usp=sharing)]
* Download DexYCB data and annotation files [[data](https://dex-ycb.github.io/)][[annotation files](https://drive.google.com/drive/folders/1pmRpgv38PXvlLOODtoxpTYnIpYTkNV6b?usp=sharing)] 


You need to follow directory structure of the `data` as below.  
```  
${ROOT}  
|-- data  
|   |-- HO3D
|   |   |-- data
|   |   |   |-- train
|   |   |   |   |-- ABF10
|   |   |   |   |-- ......
|   |   |   |-- evaluation
|   |   |   |-- annotations
|   |   |   |   |-- HO3D_train_data.json
|   |   |   |   |-- HO3D_evaluation_data.json
|   |-- DEX_YCB
|   |   |-- data
|   |   |   |-- 20200709-subject-01
|   |   |   |-- ......
|   |   |   |-- annotations
|   |   |   |   |--DEX_YCB_s0_train_data.json
|   |   |   |   |--DEX_YCB_s0_test_data.json
``` 

# Directory
## Root  
The `${ROOT}` is described as below.  
```  
${ROOT}  
|-- data  
|-- demo
|-- common  
|-- main  
|-- output  
```  
* `data` contains data loading codes and soft links to images and annotations directories.  
* `demo` contains demo codes.
* `common` contains kernel codes for InteractHand.  
* `main` contains high-level codes for training or testing the network.  
* `output` contains log, trained models, visualized outputs, and test result.  


## Output  
You need to follow the directory structure of the `output` folder as below.  
```  
${ROOT}  
|-- output  
|   |-- log  
|   |-- model_dump  
|   |-- result  
|   |-- vis  
```  
* Creating `output` folder as soft link form is recommended instead of folder form because it would take large storage capacity.  
* `log` folder contains training log file.  
* `model_dump` folder contains saved checkpoints for each epoch.  
* `result` folder contains final estimation files generated in the testing stage.  
* `vis` folder contains visualized results.  

# Running InteractHand

## Train  
In the `main` folder, set trainset in `config.py` (as 'HO3D' or 'DEX_YCB') and run  
```bash  
python train.py --gpu 0-3
```  
to train InteractHand on the GPU 0,1,2,3. `--gpu 0,1,2,3` can be used instead of `--gpu 0-3`.

## Test  
Place trained model at the `output/model_dump/`.
  
In the `main` folder, set testset in `config.py` (as 'HO3D' or 'DEX_YCB') and run  
```bash  
python test.py --gpu 0-3 --test_epoch {test epoch}  
```  
