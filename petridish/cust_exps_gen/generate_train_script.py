"""
This script generate a training script given
a net_info stored in a single line in a text file,
and a data-set (cifar10, cifar100 or ilsvrc).

Typical usage:

python3 petridish/cust_exps_gen/generate_train_script.py \
    --net_info_fn=<path_to_net_info_str_fn> \
    --ds_name=cifar10

This will result in a script that you can run to train
your model. The default location is scripts/custom.sh
You can change the destination with --out_fn.

If you have different nr_gpu, you can change it in the resulting
script, or you can modify this script to change the `options` directly
before it is written out to a file.
"""

import argparse, os

from petridish.cust_exps_gen.util import (
    form_cust_exp_bash, generate_train_options,
    cifar_default_search_options, cifar_default_train_options,
    imagenet_mobile_default_train_options, model_desc_fn_to_option,
    options_to_script_str
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_info_fn',
        type=str, help='path to net_info_fn')
    parser.add_argument('--ds_name',
        type=str, help='dataset type for final training: cifar10, cifar100, or ilsvrc')
    parser.add_argument('--name',
        type=str, help='experiment index',
        default='custom')
    parser.add_argument('--out_fn',
        type=str, help='output file path',
        default=None)

    args = parser.parse_args()

    options, desc = generate_train_options(args.net_info_fn, args.ds_name)
    script_str = options_to_script_str(options)
    out_fn = args.out_fn if args.out_fn else \
        os.path.join('scripts', '{}.sh'.format(args.name))
    print(' {} : {}'.format(out_fn, desc))
    with open(out_fn, 'wt') as fout:
        fout.write(script_str)
        fout.write('\n# {}\n'.format(desc))
