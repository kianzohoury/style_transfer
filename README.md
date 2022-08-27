# Neural Style Transfer
A PyTorch implementation of [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
with additional features.

<p float="left">
    <img src="examples/Tuebingen_Neckarfront.jpeg" width="240" height="160"/>
    <img src="examples/kandinsky.jpg" width="240" height="160"/>
    <img src="examples/shipwreck.jpg" width="240" height="160"/>
</p>

<p float="left">
    <img src="examples/scream.jpg" width="240" height="160"/>
    <img src="examples/picasso.jpg" width="240" height="160"/>
    <img src="examples/great_wave.jpg" width="240" height="160"/>
</p>

## Install
Clone the repo to install:
```
https://github.com/kianzohoury/style_transfer.git
```

## Usage
```
$ python3 style_transfer path/to/content_img path/to/style_img
--save path/to/output \
--iters 300 \
--beta 0.0001 \
--lr 1.0 \
--tv-reg \
--device cuda
```
