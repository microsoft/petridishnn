# This module contains the python script entries of the petridish.
# 1. the server handles search logics such as which to expand,
# and which to train and evaluate. The critic for evaluating models
# are in petridish/nas_control/critic. It is mostly not used now
# as we use FILO.
# 2. the trainer handles training/evaluation of the models
# (Part of this is in the petridish/model/train_eval file/module)
# 3. local crawler organizes a pool of trainers.