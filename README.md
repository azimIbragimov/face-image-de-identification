# Face Image De-Identification By Feature Space Adversarial Perturbation - Unofficial PyTorch Implementation
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" height=22.5></a>  
![issueBadge](https://img.shields.io/github/issues/azimibragimov/face-image-de-identification)   ![starBadge](https://img.shields.io/github/stars/azimibragimov/face-image-de-identification)   ![repoSize](https://img.shields.io/github/repo-size/azimibragimov/face-image-de-identification) 

> Privacy leakage in images attracts increasing concerns these days, as photos uploaded to large social platforms are usually not processed by proper privacy protection mechanisms. Moreover, with advanced artificial intelligence (AI) tools such as deep neural network (DNN), an adversary can detect people's identities and collect other sensitive personal information from images at an unprecedented scale. In this paper, we introduce a novel face image de‐identification framework using adversarial perturbations in the feature space. Manipulating the feature space vector ensures the good transferability of our framework. Moreover, the proposed feature space adversarial perturbation generation algorithm can successfully protect the identity‐related information while ensuring the other attributes remain similar. Finally, we conduct extensive experiments on two face image datasets to evaluate the performance of the proposed method. Our results show that the proposed method can generate real‐looking privacy‐preserving images efficiently. Although our framework has only been tested on two real‐life face image datasets, it can be easily extended to other types of images.

## Description   
Unnoficial implementation of [Face Image De-Identification By Feature Space Adversarial Perturbation](https://onlinelibrary.wiley.com/doi/epdf/10.1002/cpe.7554) with Pytorch


<p align="center">
<img src="assets/comparison.png" width="800px"/>
<br>
The proposed re-implementation can de-identify face images using adversarial perturbations in the feature space. The top row represents source images, the middle row shows results shown in the original paper, and the last row shows the results of our re-implementation. 
</p>

## Getting Started
### Installation
- Clone this repo:
``` 
git clone https://github.com/azimIbragimov/face-image-de-identification.git
cd face-image-de-identification
```
- Dependencies:  
We recommend running this repository using [Docker](//www.docker.com/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). 
All dependencies for defining the environment are provided in the Dockerfile

Once you have installed Docker on your system, run the following CLI commands to install the image:
`
docker build -t face-image-de-identification .
`
Congratulations, you have installed the image. Each time you want to access the image, run th following command
`
docker run --gpus all -v ./workspace -it face-image-deidentification
cd /workspace
`
### Required pretrained Models
Please download the pre-trained models from the following links. Each pSp model contains the entire pSp architecture, including the encoder and decoder weights.
| Path | Description
| :--- | :----------
|[StyleGAN Inversion](https://drive.google.com/file/d/1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0/view?usp=sharing)  | pSp trained with the FFHQ dataset for StyleGAN inversion.
|[ArcFace Resnet](https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&cid=4A83B6B633B029CC&id=4A83B6B633B029CC%215650&parId=4A83B6B633B029CC%215581&o=OneUp) | Backbone ArcFace Resnet 50 model

Note: You must place these models in the root directory (In other words, place the model in the same folder where `Dockerfile` is located).

## Inference
We define the following interface to interact with the codebase. 
For example, 
```
python inference.py \
	--img /path/to/img.png # Note: Image file can have any extension (.png, .jpg, .jpeg, .bmp, ...)
	--id_loss 0.8 # range [0, 2]
	--out /path/to/output.png
```



## Credits
**Pixel2Style2Pixel implementation:**
https://github.com/eladrich/pixel2style2pixel
Copyright (c) 2020 Elad Richardson, Yuval Alaluf
License (MIT) https://github.com/rosinality/stylegan2-pytorch/blob/master/LICENSE

**InsightFace Resnet 50 implementation:**
https://github.com/deepinsight/insightface


**StyleGAN2 implementation:**  
https://github.com/rosinality/stylegan2-pytorch  
Copyright (c) 2019 Kim Seonghyeon  
License (MIT) https://github.com/rosinality/stylegan2-pytorch/blob/master/LICENSE  

**MTCNN, IR-SE50, and ArcFace models and implementations:**  
https://github.com/TreB1eN/InsightFace_Pytorch  
Copyright (c) 2018 TreB1eN  
License (MIT) https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/LICENSE  

**CurricularFace model and implementation:**   
https://github.com/HuangYG123/CurricularFace  
Copyright (c) 2020 HuangYG123  
License (MIT) https://github.com/HuangYG123/CurricularFace/blob/master/LICENSE  

**Ranger optimizer implementation:**  
https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer   
License (Apache License 2.0) https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer/blob/master/LICENSE  

**LPIPS implementation:**  
https://github.com/S-aiueo32/lpips-pytorch  
Copyright (c) 2020, Sou Uchida  
License (BSD 2-Clause) https://github.com/S-aiueo32/lpips-pytorch/blob/master/LICENSE  

**Please Note**: The CUDA files under the [StyleGAN2 ops directory](https://github.com/eladrich/pixel2style2pixel/tree/master/models/stylegan2/op) are made available under the [Nvidia Source Code License-NC](https://nvlabs.github.io/stylegan2/license.html)

## Inspired by pSp
Below are several works inspired by pSp that we found particularly interesting:  

**Reverse Toonification**  
Using our pSp encoder, artist [Nathan Shipley](https://linktr.ee/nathan_shipley) transformed animated figures and paintings into real life. Check out his amazing work on his [twitter page](https://twitter.com/citizenplain?lang=en) and [website](http://www.nathanshipley.com/gan).   

**Deploying pSp with StyleSpace for Editing**  
Awesome work from [Justin Pinkney](https://www.justinpinkney.com/) who deployed our pSp model on Runway and provided support for editing the resulting inversions using the [StyleSpace Analysis paper](https://arxiv.org/abs/2011.12799). Check out his repository [here](https://github.com/justinpinkney/pixel2style2pixel).

**Encoder4Editing (e4e)**   
Building on the work of pSp, Tov et al. design an encoder to enable high quality edits on real images. Check out their [paper](https://arxiv.org/abs/2102.02766) and [code](https://github.com/omertov/encoder4editing).

**Style-based Age Manipulation (SAM)**  
Leveraging pSp and the rich semantics of StyleGAN, SAM learns non-linear latent space paths for modeling the age transformation of real face images. Check out the project page [here](https://yuval-alaluf.github.io/SAM/).

**ReStyle**  
ReStyle builds on recent encoders such as pSp and e4e by introducing an iterative refinment mechanism to gradually improve the inversion of real images. Check out the project page [here](https://yuval-alaluf.github.io/restyle-encoder/).

## pSp in the Media
* bycloud: [AI Generates Cartoon Characters In Real Life Pixel2Style2Pixel](https://www.youtube.com/watch?v=g-N8lfceclI&ab_channel=bycloud)
* Synced: [Pixel2Style2Pixel: Novel Encoder Architecture Boosts Facial Image-To-Image Translation](https://syncedreview.com/2020/08/07/pixel2style2pixel-novel-encoder-architecture-boosts-facial-image-to-image-translation/)
* Cartoon Brew: [An Artist Has Used Machine Learning To Turn Animated Characters Into Creepy Photorealistic Figures](https://www.cartoonbrew.com/tech/an-artist-has-used-machine-learning-to-turn-animated-characters-into-creepy-photorealistic-figures-197975.html)


## Citation
If you use this code for your research, please cite our paper <a href="https://arxiv.org/abs/2008.00951">Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation</a>:

```
@InProceedings{richardson2021encoding,
      author = {Richardson, Elad and Alaluf, Yuval and Patashnik, Or and Nitzan, Yotam and Azar, Yaniv and Shapiro, Stav and Cohen-Or, Daniel},
      title = {Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation},
      booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month = {June},
      year = {2021}
}
```
