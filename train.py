
import os


from argparse import ArgumentParser
from .style_transfer import run_gatys_style_transfer


def model_fn(model_dir):
    pass


def main():
    parser = ArgumentParser()

    # training hyper-parameters
    parser.add_argument('--image-size', type=tuple, default=(512, 512))
    parser.add_argument('--normalize-input', type=bool, default=True)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0e-6)
    parser.add_argument('--num-iters', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--tv-reg', type=float, default=1.0e-6)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--random', type=bool, default=False)

    # train/test and output paths
    parser.add_argument(
        '--save-path', type=str, default=os.environ['SM_OUTPUT_DATA_DIR']
    )
    # parser.add_argument(
    #     '--model-dir', type=str, default=os.environ['SM_MODEL_DIR']
    # )
    parser.add_argument(
        '--content-src', type=str, default=os.environ['SM_CHANNEL_CONTENT']
    )
    parser.add_argument(
        '--style-src', type=str, default=os.environ['SM_CHANNEL_STYLE']
    )
    parser.add_argument(
        '--display', type=bool, default=True
    )

    # parse CL args
    args = parser.parse_args()
    # run gatys training
    run_gatys_style_transfer(**args.__dict__)


if __name__ == "__main__":
    main()



