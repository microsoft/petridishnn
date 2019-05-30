# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script inspects a log.log file of a search in order
to extract the model architecture files for final training.
The key to find in the log.log files is l_mi=[xx,xxx,xxx], which
contains the list of model iter of the models on the perf convex hull
during search.

The most typical usage is

python3 petridish/analysis/search.py \
    --log_fn=<search log.log> \
    --max_mi=75 \
    --do_save  \

where max_mi determines which mi are allowed, and do_save is used
to actually write down the found models. (Run the command first without do_save to
gather some info first, and once you are sure that you are right, then enable do_save)
"""



import numpy as np
import json
import os, re, sys
import argparse
import subprocess

from petridish.analysis.search_util import (
    inspect_one_model, inspect_models, convex_hull_model_iters
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_fn', type=str, default=None,
        help=('Petridish search log paths in a comma separated list.'
             ' The logs are from the same run.'
             ' There are multiple logs b/c of interruption/recovery.')
    )
    parser.add_argument(
        '--mi', type=str, default=None,
        help=('Model indices, in a comma separated list.'
            ' If this is not specified, we will use the list on the convex hull')
    )
    parser.add_argument(
        '--out_fn', type=str, default=None,
        help=(
            'Output fns in a comma separated list; if it is not specified, '
            'then a default one based on log_fn is generated for you')
    )
    parser.add_argument(
        '--do_save', default=False, action='store_true'
    )
    parser.add_argument(
        '--max_mi', type=int, default=None,
        help='The maximum model iter value allowed. This stops the search early artificially.'
    )
    args = parser.parse_args()

    l_in_fn = args.log_fn.split(',')
    if args.mi:
        l_mi = [int(x) for x in args.mi.split(',')]
    else:
        l_mi = convex_hull_model_iters(l_in_fn, args.max_mi, verbose=True)
    inspect_models(l_in_fn, l_mi, args.out_fn, args.do_save, verbose=True)

