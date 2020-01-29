#!/bin/bash

DATE=`date '+%Y_%m_%d_%H_%M_%S'`

log_dir=${GLOBAL_LOG_DIR}/petridish_cifar_lstm_$DATE
model_dir=${GLOBAL_MODEL_DIR}/petridish_cifar_lstm_$DATE

mkdir -p $model_dir
mkdir -p $log_dir

python petridish_main.py \
--ds_name=cifar10 \
--data_dir=${GLOBAL_DATA_DIR} \
--log_dir=$log_dir \
--model_dir=$model_dir \
--batch_size=128 \
--nr_gpu=1 \
--input_size=32 \
--init_channel=16 \
--s_type=basic \
--b_type=basic \
--prediction_feature=1x1 \
--ls_method=100 \
--max_growth=20 \
--init_n=3 \
--do_factorized_reduction \
--controller_type=0 \
--critic_type=1 \
--controller_max_depth=128 \
--controller_dropout_kp=0.5 \
--controller_batch_size=1 \
--controller_train_every=2 \
--controller_sort_every=2 \
--lstm_size=32 \
--num_lstms=2 \
--n_changes_per_hallucination=1 \
--n_hallucinations_per_node=2 \
--debug_child_max_epoch=40 \
> $log_dir/stdout.txt



# options to try:
#--do_factorized_reduction \
#--do_multiple_skips \
#--use_cosine_lr_schedule \
#--