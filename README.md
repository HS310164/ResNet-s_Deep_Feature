# The Model Extracting Deep Feature
This is a pytorch code to extract Deep Feature from videos.  
This code is written for Python3.

## Requirements

This code has the following requirements

### Python Libraries

- PyTorch 0.4 or later (tested by 0.4.1)
- torchvision 0.2 or later (tested by 0.2.1)
- Numpy (tested by 1.15.3)
- Pillow (tested by 5.3.0)
- OpenCV (tested by 3.4.3.18)
- tqdm (tested by 4.28.1)

### The other Libraries

- FFmpeg (tested by 3.4.4-1)

## About Model

[ResNet-152](https://arxiv.org/abs/1512.03385) Pretrained [ImageNet](http://www.image-net.org) (Trained by Pytorch official)

## Deep Feature

This code gets last Average Pooling output from each frames.  
Outputs is 2048 dimensional vectors.

## Usage

```
python main.py --input <input_dir> --output <output_dir> [ --only_hand]
```

- input_dir

Put videos this directory to extract Deep Feature

- output_dir

Outputs Deep Feature is saved files into this directory

- only_hand

If this option is true, ectract deep feature from neibor hands