#!/bin/bash

python petridish_main.py \
--ds_name=age_only \
--data_dir=${GLOBAL_DATA_DIR} \
--log_dir=${GLOBAL_LOG_DIR} \
--model_dir=${GLOBAL_MODEL_DIR} \
--batch_size=64 \
--nr_gpu=1 \
--init_channel=100 \
--s_type=basic \
--b_type=basic \
--prediction_feature=none \
--ls_method=100 \
--controller_type=1 \
--max_growth=10 \
--init_n=5 \
--child_train_from_scratch \
--hint_weight=0 \
--stop_input_gradient=1 \
--launch_remote_jobs \
--regularize_coef=const \
--regularize_const=0 \
--num_classes=2 \
#--optimizer=gd \