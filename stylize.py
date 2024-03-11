
import shutil
from pathlib import Path
from PIL import Image
from typing import List, Optional, Tuple, Union

import torch
import torchvision.transforms as T
from torch.optim import LBFGS


import utils
from .models import vgg, VGGNetwork, TransformationNetwork
from losses import perceptual_loss, total_variation_loss


def run_gatys_optimization(
    content_src: str,
    style_src: str,
    image_size: Union[int, Tuple[int, int]] = (512, 512),
    content_labels: Optional[Union[List[str], str]] = "default",
    style_labels: Optional[Union[List[str], str]] = "default",
    content_weight: float = 1.0,
    style_weight: float = 1e3,
    lbfgs_iters: int = 10,
    num_steps: int = 50,
    lr: float = 1e-3,
    tv_weight: float = 1e-6,
    device: str = "cpu",
    init_noise: bool = False,
    save_fp: Optional[str] = None,
    save_gif: bool = False,
    save_all: bool = False,
    save_losses: bool = False
) -> Image:
    """Runs Gatys et al.'s neural style transfer method.

    Since it performs optimization steps on an image directly, this method is
    considerably slow. It is recommended to keep the number of L-BFGS optimizer
    iterations relatively small (e.g. <= 20), as higher numbers do not
    significantly improve results.

    Args:
        content_src (str): Path to content image.
        style_src (str): Path to style image.
        image_size (tuple or int): Shape to resize images.
            Default: (512, 512).
        content_labels (list, str, optional): Layers to calculate content
            losses from. If None is specified, content representation layers
            are ignored; otherwise, default layers are chosen.
        style_labels (list, str, optional): Layers to calculate style losses
            from. If None is specified, style representation layers are
            ignored; otherwise, default layers are chosen.
        content_weight (float): Content loss weight. Default: 1.0.
        style_weight (float): Style loss weight. Default: 1e3.
        lbfgs_iters (int): Max number of L-BFGS iterations per optimization
            step. Default: 10.
        num_steps (int): Number of image optimizations, resulting in a maximum
            of `lbfgs_iters` * `num_steps` total L-BFGS iterations.
            Default: 50.
        lr (float): Learning rate for L-BFGS optimizer. Default: 1e-3.
        tv_weight (float): Total variation regularization weight.
            Default: 1e-6.
        device (str): Device. Default: 'cpu'.
        init_noise (bool): Initializes generated image with noise.
            Default: False.
        save_fp (Optional[str]): Path to save generated image, using a valid
            format (e.g. jpg, tiff). Default: None.
        save_gif (bool): If True, saves a .gif version of the image saved
            under `save_fp`. Default: False.
        save_all (bool): If True, saves a folder of all intermediate images
            in the same directory as `save_fp`. Default: False.
        save_losses (bool): If True, saves the losses at each optimization
            step as .txt file in the same directory as `save_fp`.
            Default: False

    Returns:
        Image: Final generated image.
    """

    # Load images.
    content_image = utils.load_image(
        filepath=content_src, size=list(image_size), device=device
    )
    style_image = utils.load_image(
        filepath=style_src, size=list(image_size), device=device
    )

    # check that at least content or style is being optimized
    if not (content_labels or style_labels):
        raise ValueError(
            "At least one of content_labels or style_labels must be provided."
        )

    # if only one of content/style labels are included, the optimization task
    # is similar to feature inversion.
    if isinstance(content_labels, str) and content_labels.lower() == "default":
        content_labels = set(vgg.DEFAULT_CONTENT_LAYERS)
    elif isinstance(content_labels, list) or isinstance(content_labels, tuple):
        content_labels = set(content_labels)
    else:
        content_labels = set()

    if isinstance(style_labels, str) and style_labels.lower() == "default":
        style_labels = set(vgg.DEFAULT_STYLE_LAYERS_16)
    elif isinstance(style_labels, list) or isinstance(style_labels, tuple):
        style_labels = set(style_labels)
    else:
        style_labels = set()

    # Load pretrained loss network (VGG16)
    vgg_network = VGGNetwork(
        feature_labels=content_labels | style_labels
    ).to(device)
    vgg_network = vgg_network.requires_grad_(False).eval()

    # Initialize image.
    if init_noise:
        generated_image = torch.randn(
            content_image.size(), device=device
        ).requires_grad_(True)
    else:
        generated_image = content_image.clone().requires_grad_(True)

    # Load optimizer.
    optimizer = LBFGS([generated_image], lr=lr, max_iter=lbfgs_iters)
    losses = []

    # get content and style target representations (only need to be calculated
    # once)
    content_img_features = vgg_network(content_image)
    style_img_features = vgg_network(style_image)
    content_targets, style_targets = {}, {}

    for label in content_labels:
        content_targets[label] = content_img_features[label].detach()
    for label in style_labels:
        style_targets[label] = style_img_features[label].detach()

    all_images, best_image, best_loss = [], None, float("inf")
    print("Starting style transfer using direct optimization...")
    for step in range(num_steps):
        print(
            f"Step [{step + 1}/{num_steps}], "
            f"Iterations [{step * lbfgs_iters}/{num_steps * lbfgs_iters}]"
        )
        c_loss = s_loss = tv_loss = 0

        # Define closure for the optimizer.
        def closure():
            # allows us to update losses outside of closure
            nonlocal c_loss, s_loss, tv_loss

            # Force image values to be between 0 and 1.
            with torch.no_grad():
                generated_image.clamp_(0, 1)
            optimizer.zero_grad()

            # Run forward pass to get features
            generated_features = vgg_network(generated_image)
            generated_content, generated_style = {}, {}
            for label in content_labels:
                generated_content[label] = generated_features[label]
            for label in style_labels:
                generated_style[label] = generated_features[label]

            # Calculate perceptual loss.
            labeled_losses = perceptual_loss(
                generated_content=generated_content,
                generated_style=generated_style,
                content_targets=content_targets,
                style_targets=style_targets,
                generated_image=generated_image,
                content_weight=content_weight,
                style_weight=style_weight,
                tv_weight=tv_weight
            )

            loss = labeled_losses["perceptual"]
            c_loss += labeled_losses["content"]
            s_loss += labeled_losses["style"]
            tv_loss += labeled_losses["tv"]

            loss.backward()
            return loss.item()

        # Optimization step.
        curr_loss = optimizer.step(closure)
        losses.append(curr_loss)

        # Display losses.
        print(
            f"Perceptual Loss: {curr_loss:.12f}\n"
            f"Content Loss: {c_loss:.12f}\n"
            f"Style Loss: {s_loss:.12f}\n"
            f"TV Loss: {tv_loss:.12f}\n"
        )

        # Save best image.
        if curr_loss < best_loss:
            best_loss = curr_loss
            best_image = utils.tensor_to_image(
                generated_image.detach().clone()
            )

        if save_all or save_gif:
            all_images.append(utils.tensor_to_image(generated_image))
    print("Style transfer is complete.")
    if save_fp:
        best_image.save(save_fp)
        print(f"Saved result to {save_fp}.")
    if save_all:
        save_dir = Path(save_fp).parent / (Path(save_fp).stem + "_all")
        if save_dir.is_dir():
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True)

        # save all intermediate images
        for i in range(len(all_images)):
            all_images[i].save(
                save_dir / (Path(save_fp).stem + f"_iter_{(i + 1) * lbfgs_iters}" + Path(save_fp).suffix)
            )
        print(f"Saved all outputs to {save_dir}.")
    if save_gif:
        if not init_noise:
            starting_image = utils.tensor_to_image(content_image)
        else:
            starting_image = all_images.pop(0)
        starting_image.save(
            fp=Path(save_fp).parent / (Path(save_fp).stem + ".gif"),
            save_all=True,
            append_images=all_images + all_images[1:-1][::-1],
            duration=2500 / len(all_images),
            loop=0
        )
        print(f"Saved animation to {Path(save_fp).stem}.gif.")
    if save_losses:
        with open(Path(save_fp).parent / (Path(save_fp).stem + "_losses.txt"), mode="w") as f:
            for i in range(len(losses)):
                f.write(f"iter: {(i + 1) * lbfgs_iters}, loss: {losses[i]}\n")

    return best_image


def run_fast_style_transfer_batched(
    img_paths: str,
    out_dir: str,
    checkpoint: str,
    img_size: Optional[Tuple[int, int], int] = 512,
    device: str = "cpu",
    batch_size: int = 4
) -> None:
    """Runs fast style transfer using transformation networks (Johnson et al.)

    Args:
        img_paths (str): Paths to images to be transformed.
        out_dir (str): Path to save stylized images to.
        checkpoint (str): Checkpoint file (as a "*.pth") where the model is
            saved.
        img_size (tuple or in, optional): Size of the images. Note that this is
            required since the images will be batched. Default: 512.
        device (str). Device to run inference on. Default: 'cpu'.
        batch_size (int): Batch size. Make sure the batch size agrees with
            GPU memory capacity. Default: 4.
    """

    # get images
    img_paths = Path(img_paths)
    img_paths = list(img_paths.iterdir()) if img_paths.is_dir() else [img_paths]
    if len(img_paths) == 0:
        print("No images found!")

    content_img = []
    print(f"Found {len(img_paths)} images. Loading images...")
    for path in img_paths:
        try:
            content_img.append(
                utils.load_image(path, size=img_size, device=device)
            )
        except IOError:
            print(f"Could not load image at {path}.")
    # organize images into batches
    content_img = torch.stack(content_img, dim=0)

    # load checkpoint
    state_dict = torch.load(checkpoint, map_location=device)
    config = state_dict["config"]

    # load trained model
    model = TransformationNetwork(
        padding_mode=config.padding_mode,
        norm_type=config.norm_type,
        upsample_type=config.upsample_type,
        output_fn=config.output_fn
    ).to(device)

    # define input normalizer (basic pre-processing step)
    input_norm = T.Normalize(
        mean=utils.IMAGENET_MEAN, std=utils.IMAGENET_STD
    )

    # path to save images
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(checkpoint).stem

    num_iters = len(content_img) // batch_size
    n_images = 0
    for i in range(num_iters):
        # get batch
        img_batch = content_img[batch_size * i: batch_size * (i + 1), :, :, :]
        # get stylized images! (inference)
        stylized_img = model(input_norm(img_batch))

        for img_idx, tensor_img in enumerate(stylized_img, 0):
            pil_img = utils.tensor_to_image(tensor_img)
            suffix = Path(img_paths[img_idx]).suffix
            save_img_name = img_paths[img_idx].stem + f"_{model_name}" + suffix
            # save to output directory
            pil_img.save(out_dir / save_img_name)
            n_images += 1
    print(f"Finished. Saved images {n_images} at {img_paths}.")
