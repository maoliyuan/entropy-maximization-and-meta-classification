##### These are the origin repository's README file, please make sure you have followed these !!! #####
##### These are the origin repository's README file, please make sure you have followed these !!! #####
##### These are the origin repository's README file, please make sure you have followed these !!! #####
##### These are the origin repository's README file, please make sure you have followed these !!! #####
##### These are the origin repository's README file, please make sure you have followed these !!! #####
# Entropy Maximization and Meta Classification for Out-of-Distribution Detection in Semantic Segmentation  
  
**Abstract** Deep neural networks (DNNs) for the semantic segmentation of images are usually trained to operate on a predefined closed set of object classes. This is in contrast to the "open world" setting where DNNs are envisioned to be deployed to. From a functional safety point of view, the ability to detect so-called "out-of-distribution" (OoD) samples, i.e., objects outside of a DNN's semantic space, is crucial for many applications such as automated driving.
We present a two-step procedure for OoD detection. Firstly, we utilize samples from the COCO dataset as OoD proxy and introduce a second training objective to maximize the softmax entropy on these samples. Starting from pretrained semantic segmentation networks we re-train a number of DNNs on different in-distribution datasets and evaluate on completely disjoint OoD datasets. Secondly, we perform a transparent post-processing step to discard false positive OoD samples by so-called "meta classification". To this end, we apply linear models to a set of hand-crafted metrics derived from the DNN's softmax probabilities.
Our method contributes to safer DNNs with more reliable overall system performance.

* More details can be found in the preprint https://arxiv.org/abs/2012.06575
* Training with [Cityscapes](https://www.cityscapes-dataset.com/) and [COCO](https://cocodataset.org), evaluation with [LostAndFound](http://www.6d-vision.com/lostandfounddataset) and [Fishyscapes](https://fishyscapes.com/)
  
## Requirements  
  
This code was tested with **Python 3.6.10** and **CUDA 10.2**. The following Python packages were installed via **pip 20.2.4**, see also ```requirements.txt```: 
```  
Cython == 0.29.21  
h5py == 3.1.0  
scikit-learn == 0.23.2  
scipy == 1.5.4  
torch == 1.7.0  
torchvision == 0.8.1
pycocotools == 2.0.2
```
**Dataset preparation**: In ```preparation/prepare_coco_segmentation.py``` a preprocessing script can be found in order prepare the COCO images serving as OoD proxy for OoD training. This script basically generates binary segmentation masks for COCO images not containing any instances that could also be assigned to one of the Cityscapes (train-)classes. Execute via:
```  
python preparation/prepare_coco_segmentation.py
```
Regarding the Cityscapes dataset, the dataloader used in this repo assumes that the *labelTrainId* images are already generated according to the [official Cityscapes script](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py).

**Cython preparation**: Make sure that the Cython script ```src/metaseg/metrics.pyx``` (on the machine where the script is deployed to) is compiled. If it has not been compiled yet:  
```  
cd src/metaseg/  
python metrics_setup.py build_ext --inplace  
cd ../../  
```  
For pretrained weights, see [https://github.com/NVIDIA/semantic-segmentation/tree/sdcnet](https://github.com/NVIDIA/semantic-segmentation/tree/sdcnet) (for DeepLabv3+) and [https://github.com/lxtGH/GALD-DGCNet](https://github.com/lxtGH/GALD-DGCNet) (for DualGCNNet).
The weights after OoD training can be downloaded [here for DeepLabv3+](https://uni-wuppertal.sciebo.de/s/kCgnr0LQuTbrArA/download) and [here for DualGCNNet](https://uni-wuppertal.sciebo.de/s/VAXiKxZ21eAF68q/download).
  
## Quick start  
  
Modify settings in ```config.py```. All files will be saved in the directory defined via ```io_root``` (Different roots for each datasets that is used). Then run:  
```  
python ood_training.py  
python meta_classification.py  
python evaluation.py  
```  
  
## More options
  
For better automation of experiments,  **command-line options** for ```ood_training.py```, ```meta_classification.py``` and ```evaluation.py``` are available.  
  
Use the ```[-h]``` argument for details about which parameters in ```config.py``` can be modified via command line. Example:  
```  
python ood_training.py -h  
```  
  
If no command-line options are provided, the settings in ```config.py``` are applied.

##### These are part of our code and how to use it !!!! #####
##### These are part of our code and how to use it !!!! #####
##### These are part of our code and how to use it !!!! #####
##### These are part of our code and how to use it !!!! #####
##### These are part of our code and how to use it !!!! #####
More preparation:
pip install -U scikit-learn

## Augmented data path
All the augmmented data are stored in ./datasets/cs_coco_embedding, but before generating the augmented data please put cropped COCO images in this directory ./datasets/cropped_coco , and run python src/dataset/cs_coco_embedding.py to generate augmented data.

## Train with augmented data
Please choose the ratio of augmented data that you want to mix with origin data. embedding_img_interval in config.py indicates the frequency of origin data that replaced by augmented data, i.e. embedding_img_interval=1 means every origin data would be replaced by augmented data.

## optim target
We support another training method with max logit, which might be helpful if you want to use this model as the front-end model of SML algorithm.
if you want to train this kind of model, simply change optim_target in config.py from 'entropy' to 'logit'

## OODMC parameters
Here are the meanings of all the parameters of OODMC and how to correctly use them to run OODMC
moment_num              default 128         corresponds to the number of samples you want to use to calculate statistic information of logits
svm_points_num          default 5e5         the number of total pixels you want to train the classifier (here is SVM)
moment_order            default 4           the number of orders you want to calculate for logits' moment, each order's moment is an array of C bits(C is classs num)
moment_weight           default 0.5         since OODMC also use entropy, which dosen't belong to any order, this is coefficient that measures how much it rely on it
SVM_eval_subsize        default 100         OODMC is extremely slow if your GPU number is small (less than 4*2080Ti is small), so this is how many picture you want to analyze

Way to use OODMC
python meta_classification.py -moment -SVM
python evaluation,py -moment