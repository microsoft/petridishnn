#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,3

DATE=`date '+%Y_%m_%d_%H_%M_%S'`

log_dir=${GLOBAL_LOG_DIR}/test_openml
model_dir=${GLOBAL_MODEL_DIR}/test_openml

mkdir -p $model_dir
mkdir -p $log_dir

export CONFIG_DIR=.
export DATA_DIR=${GLOBAL_DATA_DIR}
export MODEL_DIR=$model_dir
export LOG_DIR=$log_dir

python $CONFIG_DIR/petridish_main.py \
--data_dir=$DATA_DIR \
--log_dir=$LOG_DIR \
--model_dir=$MODEL_DIR \
--adaloss_final_extra=0.0 \
--adaloss_gamma=0.07 \
--adaloss_momentum=0.9 \
--adaloss_update_per=100 \
--alter_label_activate_frac=0.75 \
--alter_loss_w=0.5 \
--b_type=basic \
--batch_norm_decay=0.9 \
--batch_size=16 \
--controller_batch_size=1 \
--controller_dropout_kp=0.5 \
--controller_max_depth=128 \
--controller_save_every=1 \
--controller_train_every=100 \
--controller_type=1 \
--critic_init_lr=0.001 \
--critic_train_epoch=40 \
--critic_type=1 \
--data_format=channels_first \
--dense_dropout_keep_prob=1.0 \
--do_factorized_reduction \
--drop_path_keep_prob=1.0 \
--ds_name=openml_3 \
--evaluate="" \
--grow_petridish_version=mp \
--hallu_final_merge_op=11 \
--high_temperature=1.0 \
--hint_weight=0 \
--init_channel=32 \
--init_lr=0.01 \
--init_n=3 \
--input_size=32 \
--input_type=float32 \
--job_type=main \
--label_smoothing=0.0 \
--launch_local_crawler \
--ls_method=100 \
--lstm_size=32 \
--max_exploration=1000 \
--max_growth=20 \
--max_train_model_epoch=80 \
--n_critic_procs=1 \
--n_greed_select_per_init=0 \
--n_hallu_procs=1 \
--n_hallus_per_init=3 \
--n_hallus_per_select=3 \
--n_model_procs=2 \
--n_parent_reuses=-1 \
--n_rand_select_per_init=1 \
--netmorph_method=soft \
--nr_gpu=1 \
--num_classes=10 \
--num_init_use_all_gpu=0 \
--num_lstms=2 \
--output_type=int32 \
--prediction_feature=bn \
--q_child_method=5 \
--q_hallu_method=5 \
--q_parent_method=7 \
--regularize_coef=const \
--regularize_const=0.0001 \
--s_type=basic \
--search_cell_based \
--sgd_moment=0.9 \
--stem_channel_rate=1 \
--w_init=var_scale \
--debug_steps_per_epoch=80 \
--debug_child_max_epoch=1 \
> ${LOG_DIR}/stdout.txt

