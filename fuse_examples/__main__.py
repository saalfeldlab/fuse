import argparse

from .test_interpolate_displacement_field import configure_parser as ra_configure_parser, test_interpolate_displacement_field
from .visualize_augmentations_paintera import configure_parser as va_configure_parser, visualize_augmentations_paintera
from .visualize_original_elastic_augment import configure_parser as ve_configure_parser, visualize_original_elastic_augment


def _main():
    parser = argparse.ArgumentParser('Fuse Examples')
    sub_parsers = parser.add_subparsers(
        title='command',
        help='run command',
        dest='command',
        required=True)
    ra_configure_parser(sub_parsers.add_parser('test-interpolate'))
    va_configure_parser(sub_parsers.add_parser('visualize-augmentations'))
    ve_configure_parser(sub_parsers.add_parser('visualize-elastic-augmentations'))

    mapping = {
        'test-interpolate': test_interpolate_displacement_field,
        'visualize-augmentations': visualize_augmentations_paintera,
        'visualize-elastic-augmentations': visualize_original_elastic_augment
    }

    args = parser.parse_args()
    print(args)

    mapping[args.command](args)


if __name__ == "__main__":
    _main()
