#!/bin/bash

DATE=`date '+%Y_%m_%d_%H_%M_%S'`

log_dir=${GLOBAL_LOG_DIR}/petridish_cifar_remote_child
model_dir=${GLOBAL_MODEL_DIR}/petridish_cifar_remote_child

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
--job_type=remote_child \
--layer_info_list_str='{"operations": [], "inputs": [], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 0}=^_^={"operations": [2], "inputs": [0], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 1}=^_^={"operations": [2], "inputs": [1], "down_sampling": 0, "aux_weight": 0.0, "is_candidate": 0, "stop_gradient": 0, "id": 2}=^_^={"operations": [2], "inputs": [2], "down_sampling": 0, "aux_weight": 0.0, "is_candidate": 0, "stop_gradient": 0, "id": 3}=^_^={"operations": [8, 1, 10], "inputs": [3, 0], "down_sampling": 0, "aux_weight": 0.0, "is_candidate": 0, "stop_gradient": 0, "id": 10}=^_^={"operations": [8, 1, 10], "inputs": [10, 1], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 18}=^_^={"operations": [14, 17, 11], "inputs": [18, 10], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 19}=^_^={"operations": [14, 17, 11], "inputs": [19, 3], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 11}=^_^={"operations": [2], "inputs": [11], "down_sampling": 1, "aux_weight": 0.0, "is_candidate": 0, "stop_gradient": 0, "id": 4}=^_^={"operations": [2], "inputs": [4], "down_sampling": 0, "aux_weight": 0.0, "is_candidate": 0, "stop_gradient": 0, "id": 5}=^_^={"operations": [5, 1, 10], "inputs": [5, 2], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 26}=^_^={"operations": [14, 17, 11], "inputs": [26, 5], "down_sampling": 0, "aux_weight": 0.0, "is_candidate": 0, "stop_gradient": 0, "id": 27}=^_^={"operations": [2], "inputs": [27], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 6}=^_^={"operations": [2], "inputs": [6], "down_sampling": 1, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 7}=^_^={"operations": [2], "inputs": [7], "down_sampling": 0, "aux_weight": 0.0, "is_candidate": 0, "stop_gradient": 0, "id": 8}=^_^={"operations": [9, 1, 10], "inputs": [8, 11], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 20}=^_^={"operations": [14, 17, 11], "inputs": [20, 8], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 21}=^_^={"operations": [2], "inputs": [21], "down_sampling": 0, "aux_weight": 0.0, "is_candidate": 0, "stop_gradient": 0, "id": 9}=^_^={"operations": [5, 1, 10], "inputs": [9, 7], "down_sampling": 0, "aux_weight": 0.0, "is_candidate": 0, "stop_gradient": 0, "id": 12}=^_^={"operations": [4, 1, 10], "inputs": [12, 27], "down_sampling": 0, "aux_weight": 0.0, "is_candidate": 0, "stop_gradient": 0, "id": 14}=^_^={"operations": [9, 1, 10], "inputs": [14, 11], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 22}=^_^={"operations": [14, 17, 11], "inputs": [22, 14], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 23}=^_^={"operations": [14, 17, 11], "inputs": [23, 12], "down_sampling": 0, "aux_weight": 0.0, "is_candidate": 0, "stop_gradient": 0, "id": 15}=^_^={"operations": [8, 1, 10], "inputs": [15, 19], "down_sampling": 0, "aux_weight": 0.0, "is_candidate": 0, "stop_gradient": 0, "id": 16}=^_^={"operations": [9, 1, 10], "inputs": [16, 15], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 24}=^_^={"operations": [14, 17, 11], "inputs": [24, 16], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 25}=^_^={"operations": [14, 17, 11], "inputs": [25, 15], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 17}=^_^={"operations": [14, 17, 11], "inputs": [17, 9], "down_sampling": 0, "aux_weight": 1.0, "is_candidate": 0, "stop_gradient": 0, "id": 13}' \
> $log_dir/stdout.txt



# options to try:
#--do_factorized_reduction \
#--do_multiple_skips \
#--use_cosine_lr_schedule \
#--