
from style_transfer import run_gatys_style_transfer
from argparse import ArgumentParser

from inspect import signature, Signature


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Main script."
    )
    subparsers = parser.add_subparsers(dest="method")

    # Define gatys method parser.
    gatys_parser = subparsers.add_parser(name="gatys")
    gatys_parameters = signature(run_gatys_style_transfer).parameters
    for param in gatys_parameters.values():
        required = param.default is Signature.empty
        param_type = str if param.annotation is Signature.empty else param.annotation
        param_default = {} if param.default is Signature.empty else param.default
        gatys_parser.add_argument(
            "--" + param.name.replace("_", "-"),
            default=param_default,
            # type=param.annotation,
            required=required and param.name != "kwargs",
            action="store_true" if param_type is bool else "store"
        )
    # Parse args
    args = parser.parse_args()

    if args.method == "gatys":
        filtered_args = {}
        for param, val in args.__dict__.items():
            if param in gatys_parameters.keys():
                filtered_args[param] = val
        run_gatys_style_transfer(**filtered_args)
    else:
        pass
