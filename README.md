# Neural Style Transfer
A PyTorch implementation of [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
with additional features.

<p float="left" style="margin: 0 auto;">
    <img src="examples/content/tuebingen_neckarfront.jpeg" width="240" height="160"/>
    <img src="examples/transfers/kandinsky.jpg" width="240" height="160"/>
    <img src="examples/transfers/shipwreck.jpg" width="240" height="160"/>
</p>

<p float="left" style="margin: 0 auto;">
    <img src="examples/transfers/scream.jpg" width="240" height="160"/>
    <img src="examples/transfers/picasso.jpg" width="240" height="160"/>
    <img src="examples/transfers/great_wave.jpg" width="240" height="160"/>
</p>

## Background
Style transfer is the process of using neural networks to transform images
into new images of a different style while still resembling the content of the
original image.

## Install
Clone the repo to install:
```
$ git clone https://github.com/kianzohoury/style_transfer.git
```

## Usage

```python
from style_transfer import run_gatys_style_transfer

output = run_gatys_style_transfer(
    content_src="henriette_mayer_van_den_bergh.jpeg",
    style_src="wassily_kandinsky_composition_VII.jpeg",
    save_path="my_style_transfer.jpg",
    image_size=(768, 640),
    num_iters=100,
    tv_reg=1e-3,
    device="cuda"
)
```


```
$ python3 style_transfer path/to/content_img path/to/style_img
--save path/to/output \
--iters 300 \
--beta 0.0001 \
--lr 1.0 \
--tv-reg \
--device cuda
```

