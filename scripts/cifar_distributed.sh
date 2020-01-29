#!/bin/bash

DATE=`date '+%Y_%m_%d_%H_%M_%S'`

export CONFIG_DIR=/home/hanzhang/code/petridishnn
export DATA_DIR=${GLOBAL_DATA_DIR}
export LOG_DIR=${GLOBAL_LOG_DIR}/petridish_cifar_distributed
export MODEL_DIR=${GLOBAL_MODEL_DIR}/petridish_cifar_distributed

mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR


python ${CONFIG_DIR}/petridish_main.py \
--ds_name=cifar100 \
--data_dir=${DATA_DIR} \
--log_dir=${LOG_DIR} \
--model_dir=${MODEL_DIR} \
--batch_size=512 \
--nr_gpu=4 \
--input_size=32 \
--init_channel=16 \
--s_type=basic \
--b_type=basic \
--prediction_feature=1x1 \
--ls_method=100 \
--max_growth=20 \
--max_exploration=10000 \
--init_n=3 \
--do_factorized_reduction \
--controller_type=0 \
--critic_type=1 \
--controller_max_depth=128 \
--controller_dropout_kp=0.5 \
--controller_batch_size=1 \
--controller_train_every=10000 \
--lstm_size=32 \
--num_lstms=2 \
--n_hallus_per_init=9 \
--n_hallus_per_select=3 \
--n_selects_per_init=32 \
--n_parent_reuses=-1 \
--q_parent_method=1 \
--q_hallu_method=1 \
--q_child_method=1 \
--job_type=main \
--grow_petridish_version=mp \
--n_hallu_procs=4 \
--n_model_procs=8 \
--n_critic_procs=1 \
--launch_local_crawler \
> $LOG_DIR/stdout.txt