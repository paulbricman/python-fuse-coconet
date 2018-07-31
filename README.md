![image](https://github.com/paubric/python-fuse-coconet/blob/master/images/fig1.png)

# Functional Storage Encoding - CoCoNet

We propose a __deep neural network__ approach for mapping the 2D pixel coordinates in an image to the corresponding Red-Green-Blue (RGB) color values. The neural network is termed CocoNet, i.e. COordinates-to-COlor NETwork. During the training process, the neural network learns to encode the input image within its layers. More specifically, the network learns __a continuous function that approximates the discrete RGB values sampled over the discrete 2D pixel locations__. At test time, given a 2D pixel coordinate, the neural network will output the approximate RGB values of the corresponding pixel. By considering every 2D pixel location, the network can actually __reconstruct the entire learned image__. It is important to note that we have to train an individual neural network for each input image, i.e. one network encodes a single image only. Our neural image encoding approach has various low-level image processing applications ranging from __image encoding, image compression and image denoising to image resampling and image completion__. We conduct experiments that include both quantitative and qualitative results, demonstrating the utility of our approach and its superiority over standard baselines, e.g. bilateral filtering or bicubic interpolation.

[Presentation](https://docs.google.com/presentation/d/1Le9Qo_bpHdKLYXZhZpf9lXUlnp9uigknEMXxGal4xvE/edit?usp=sharing)

# Installation

The demontration script are written Python 3 using Keras with Tensorflow back-end, along with other utility libraries. 

## Linux
Install Python 3.
```
sudo apt-get install python3.6
```
Install TKinter.
```
apt-get install python-tk
```
Install python module requirements from provided text file.
```
pip install -r requirements.txt
```
Run test file.
```
python3 test.py
```

## Windows and Mac OS X
Install Python 3 and TKinter.
```
good luck
```
Install python module requirements from provided text file.
```
pip install -r requirements.txt
```
Run test file.
```
python3 test.py
```
## Docker version
[Install Docker](https://docs.docker.com/install/#releases)
Build Docker image.
```
sudo make bash GPU=0
```
Install additional stuff.
```
apt-get install python-tk
```
Clone repository.
Install python module requirements from provided text file.
```
pip install -r requirements.txt
```
Run test file.
```
python3 test.py
```
