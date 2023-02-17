# alexnet-pytorch

***

This is an implementaiton of AlexNet, as introduced in the paper "[ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)" by Alex Krizhevsky et al. 

## Prerequisites

***

* python>=3.9.12

* torch==1.13.1

* torchvision==0.14.1

You can install required packages by:

```bash
pip3 install -r requirements.txt
```

## Training

```bash
python3 train.py
```

Specify `TRAIN_IMG_DIR` in the script before training.

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.