# Functional Storage Encoding - CoCoNet

We propose a deep neural network approach for mapping the 2D pixel coordinates in an image to the corresponding Red-Green-Blue (RGB) color values. The neural network is termed CocoNet, i.e. COordinates-to-COlor NETwork. During the training process, the neural network learns to encode the input image within its layers. More specifically, the network learns a continuous function that approximates the discrete RGB values sampled over the discrete 2D pixel locations. At test time, given a 2D pixel coordinate, the neural network will output the approximate RGB values of the corresponding pixel. By considering every 2D pixel location, the network can actually reconstruct the entire learned image. It is important to note that we have to train an individual neural network for each input image, i.e. one network encodes a single image only. To the best of our knowledge, we are the first to propose a neural approach for encoding images individually, by learning a mapping from the 2D pixel coordinate space to the RGB color space. Our neural image encoding approach has various low-level image processing applications ranging from image encoding, image compression and image denoising to image resampling and image completion. We conduct experiments that include both quantitative and qualitative results, demonstrating the utility of our approach and its superiority over standard baselines, e.g. bilateral filtering or bicubic interpolation.

# Installation

```
sudo apt-get install python3.6
apt-get install python-tk
pip install -r requirements.txt
```
