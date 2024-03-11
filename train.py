
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from .losses import perceptual_loss, total_variation_loss
from .models import TransformationNetwork, VGGNetwork


def train_transformation_network(config):
    """Runs training for transformation networks given a training config."""
    # use of CPU for training is strongly discouraged (however, for inference
    # it is fine)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load style image
    style_src = config.style
    style_image = utils.load_image(
        style_src,
        size=config.img_size,
        device=device
    )

    # load training set (replace with own dataset if you don't want to use
    # fiftyone zoo or MS COCO)
    coco_dataset = utils.load_coco_zoo_dataset(
        root_dir=config.data_dir,
        max_samples=config.max_samples,
        img_size=config.img_size
    )
    coco_dataset.choose_split("train")

    train_loader = DataLoader(
        dataset=coco_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=True,
        persistent_workers=False
    )

    # load trainable transformation network
    model = TransformationNetwork(
        padding_mode=config.padding_mode,
        norm_type=config.norm_type,
        upsample_type=config.upsample_type,
        output_fn=None
    ).to(device)

    # optimizer (strongly recommend Adam/AdamW)
    optim = AdamW(model.parameters(), lr=config.lr)

    # Load pretrained loss network
    loss_network = VGGNetwork(
        model_type=config.vgg,
        use_avg_pool=config.use_avg_pool,
        feature_labels=config.content_labels + config.style_labels
    ).requires_grad_(False).to(device).eval()

    # resume training if folder already exists (note: will load model from
    # the best previous checkpoint)
    save_dir = Path(config.save_dir)
    checkpoint_name = save_dir.stem + ".pth"
    if save_dir.is_dir() and save_dir.exists():
        state_dict = torch.load(checkpoint_name, map_location=device)
        model.load_state_dict(state_dict["model"])
        optim.load_state_dict(state_dict["optim"])
        last_epoch = state_dict["epoch"] + 1
        epoch_history = state_dict["epoch_history"]
        best_loss = state_dict["best_loss"]
        global_step = state_dict["global_step"]
    else:
        last_epoch = global_step = 0
        epoch_history = []
        best_loss = float("inf")

    # log results to tensorboard if desired
    if config.tensorboard:
        tb_writer = SummaryWriter(log_dir=save_dir / "logs")
    else:
        tb_writer = None

    # define input normalizer (basic pre-processing step)
    input_norm = T.Normalize(
        mean=utils.IMAGENET_MEAN, std=utils.IMAGENET_STD
    )

    # get style targets (note: these only need to be computed once)
    style_targets = loss_network(style_image)
    style_targets = {
        label: style_targets[label] for label in config.style_labels
    }
    # expand targets to match the batch size (broadcasting is also fine)
    for label, target in style_targets.items():
        style_targets[label] = target.repeat(config.batch_size, 1, 1, 1)

    print("Beginning training...")
    model.train()
    for epoch in range(last_epoch, last_epoch + config.epochs):
        # reset running metrics
        perceptual_loss_total = content_loss_total = \
            style_loss_total = tv_loss_total = 0
        with tqdm(train_loader) as tq:
            tq.set_description(
                f"Epoch [{epoch + 1}/{last_epoch + config.epochs}]")
            for idx, image in enumerate(tq, 0):
                # normalize image
                image = input_norm(image.to(device))

                # produce stylized image
                gen_image = model(image)

                # extract content/style features
                gen_features = loss_network(input_norm(gen_image))
                target_features = loss_network(image)

                gen_content, gen_style, content_targets, \
                    style_targets_batch = {}, {}, {}, {}

                for label in config.content_labels:
                    gen_content[label] = gen_features[label]
                    content_targets[label] = target_features[label]

                for label in config.style_labels:
                    gen_style[label] = gen_features[label]

                # trim style targets by current batch size
                bsize = image.shape[0]
                for label, target in style_targets.items():
                    style_targets_batch[label] = target[:bsize, :, :, :]

                # calculate perceptual loss
                losses = perceptual_loss(
                    generated_content=gen_content,
                    generated_style=gen_style,
                    content_targets=content_targets,
                    style_targets=style_targets_batch,
                    content_weight=config.content_weight,
                    style_weight=config.style_weight
                )

                loss = losses["perceptual"]
                if config.tv_weight > 0:
                    tv_loss = total_variation_loss(gen_image)
                else:
                    tv_loss = 0
                loss += config.tv_weight * tv_loss

                # compute gradient, run backprop, update model params
                loss.backward()
                optim.step()
                optim.zero_grad()

                # aggregate metrics
                perceptual_loss_total += loss.item()
                if config.content_weight > 0:
                    content_loss_total += losses["content_loss"].item()
                if config.style_weight > 0:
                    style_loss_total += losses["style_loss"].item()
                if config.tv_weight > 0:
                    tv_loss_total += tv_loss

                # display metrics
                tq.set_postfix({
                    "Total": round(perceptual_loss_total / (idx + 1), 6),
                    "C": round(content_loss_total / (idx + 1), 6),
                    "S": round(style_loss_total / (idx + 1), 6),
                    "TV": round(tv_loss_total / (idx + 1), 6)
                })
                global_step += 1

        # store epoch results
        epoch_history.append({
            "perceptual": perceptual_loss_total,
            "content": content_loss_total,
            "style": style_loss_total,
            "tv": tv_loss_total
        })

        # log to tensorboard
        if config.tensorboard and global_step % 100 == 0:
            tb_writer.add_scalar(
                tag="loss/perceptual",
                scalar_value=perceptual_loss_total,
                global_step=global_step
            )
            tb_writer.add_scalar(
                tag="loss/content",
                scalar_value=content_loss_total,
                global_step=global_step
            )
            tb_writer.add_scalar(
                tag="loss/style",
                scalar_value=style_loss_total,
                global_step=global_step
            )
            tb_writer.add_scalar(
                tag="loss/tv",
                scalar_value=tv_loss_total,
                global_step=global_step
            )

            # display a some images
            np.random.seed(0)
            img_indices = np.random.randint(0, len(coco_dataset), 4)
            for img_idx in img_indices:
                img = coco_dataset[img_idx].unsqueeze(0)
                gen_img = model(input_norm(img.to(device)))
                tb_writer.add_image(
                    tag=f"content/{img_idx}",
                    img_tensor=img,
                    global_step=global_step
                )
                tb_writer.add_image(
                    tag=f"generated/{img_idx}",
                    img_tensor=gen_img.detach().cpu(),
                    global_step=global_step
                )

        # save best model
        if perceptual_loss_total < best_loss:
            best_loss = perceptual_loss_total
            print("Saving best model...")
            torch.save(
                {
                    "model": model.cpu().state_dict(),
                    "optim": optim.state_dict(),
                    "epoch": epoch,
                    "epoch_history": epoch_history,
                    "best_loss": best_loss
                },
                f=checkpoint_name
            )
            # send model back to gpu
            model = model.to(device)
    print("Finished training!...")


def main():
    parser = ArgumentParser()

    parser.add_argument(
        '--style', type=str, default="examples/style/mosaic.jpg"
    )
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--img-size', type=tuple, default=(512, 512))
    parser.add_argument('--max-samples', type=int, default=1e5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1.0e-3)
    parser.add_argument('--content-weight', type=float, default=1.0)
    parser.add_argument('--style-weight', type=float, default=1e5)
    parser.add_argument('--tv-weight', type=float, default=1.0e-6)
    parser.add_argument('--anisotropic', action='store_true')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--tensorboard', action='store_true')

    # transformation network arguments
    parser.add_argument('--padding-mode', type=str, default="reflect")
    parser.add_argument('--norm-type', type=str, default="instance")
    parser.add_argument('--upsample-type', type=str, default="interpolate")
    parser.add_argument('--output-fn', type=str, default=None)

    # vgg arguments
    parser.add_argument('--vgg', type=str, default="vgg16")
    parser.add_argument('--use-avg-pool', action='store_true')
    parser.add_argument('--content-labels', type=list, default=None)
    parser.add_argument('--style-labels', type=list, default=None)

    # parse command line arguments
    training_config = parser.parse_args()

    # run training
    train_transformation_network(training_config)


if __name__ == "__main__":
    main()
