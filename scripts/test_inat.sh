!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
DATE=`date '+%Y_%m_%d_%H_%M_%S'`

log_dir=${GLOBAL_LOG_DIR}/test_inat2017_1000
model_dir=${GLOBAL_MODEL_DIR}/test_inat2017_1000

mkdir -p $model_dir
mkdir -p $log_dir

export CONFIG_DIR=.
export DATA_DIR=${GLOBAL_DATA_DIR}
export MODEL_DIR=$model_dir
export LOG_DIR=$log_dir

# Run the actual job
python $CONFIG_DIR/petridish/app/petridish_main.py \
--data_dir=$DATA_DIR \
--log_dir=$LOG_DIR \
--model_dir=$MODEL_DIR \
--adaloss_final_extra=0.0 \
--adaloss_gamma=0.07 \
--adaloss_momentum=0.9 \
--adaloss_update_per=100 \
--alter_label_activate_frac=0.75 \
--alter_loss_w=0.5 \
--b_type=bottleneck \
--batch_norm_decay=0.9 \
--batch_norm_epsilon=1e-05 \
--batch_size_per_gpu=32 \
--controller_batch_size=1 \
--controller_dropout_kp=1 \
--controller_max_depth=128 \
--controller_save_every=1 \
--controller_train_every=10000 \
--controller_type=0 \
--critic_init_lr=0.001 \
--critic_train_epoch=40 \
--critic_type=0 \
--data_format=channels_first \
--dense_dropout_keep_prob=1.0 \
--do_remote_child_inf_runner \
--drop_path_keep_prob=0.7 \
--ds_name=inat2017_1000 \
--grow_petridish_version=mp \
--hallu_final_merge_op=11 \
--high_temperature=1.0 \
--hint_weight=0 \
--init_channel=10 \
--init_lr_per_sample=0.00125 \
--input_type=float32 \
--job_type=remote_child \
--label_smoothing=0.1 \
--ls_method=0 \
--lstm_size=32 \
--max_exploration=100000 \
--max_growth=16 \
--max_train_model_epoch=250 \
--n_critic_procs_per_gpu=1 \
--n_hallu_procs_per_gpu=1 \
--n_model_procs_per_gpu=1 \
--n_parent_reuses=-1 \
--net_info_str='{"reduction": [{"operations": [], "inputs": [], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 0}, {"operations": [], "inputs": [], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 1}, {"operations": [25, 8, 24, 22], "inputs": [1, 1, 1], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "extra_dict": {"ops_ids": [2, 3, 1], "fs_omega": [7.309421297396757e-12, -6.939733076388732e-12, -6.78657867750343e-12]}, "stop_gradient": 0, "id": 3}, {"operations": [8, 25, 25, 22], "inputs": [3, 1, 3], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "extra_dict": {"ops_ids": [3, 8, 2], "fs_omega": [8.124451632285368e-12, 7.385524483649597e-12, 5.890838598171522e-12]}, "stop_gradient": 0, "id": 4}, {"operations": [8, 1, 24, 22], "inputs": [4, 4, 4], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "extra_dict": {"ops_ids": [15, 17, 13], "fs_omega": [4.73896139344121e-12, -3.5095303399512723e-12, 3.2714653468851607e-12]}, "stop_gradient": 0, "id": 5}, {"operations": [23, 1, 14, 14, 14, 11], "inputs": [1, 1, 3, 4, 5], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 2}], "master": [{"operations": [], "inputs": [], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 0}, {"operations": [], "inputs": [], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 1}, {"operations": [1, 1, "reduction"], "inputs": [1, 1], "down_sampling": 1, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 22}, {"operations": [1, 1, "reduction"], "inputs": [1, 22], "down_sampling": 1, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 23}, {"operations": [1, 1, "normal"], "inputs": [23, 23], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 2}, {"operations": [1, 1, "normal"], "inputs": [23, 2], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 3}, {"operations": [1, 1, "normal"], "inputs": [2, 3], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 4}, {"operations": [1, 1, "normal"], "inputs": [3, 4], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 5}, {"operations": [1, 1, "normal"], "inputs": [4, 5], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 6}, {"operations": [1, 1, "normal"], "inputs": [5, 6], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 7}, {"operations": [1, 1, "reduction"], "inputs": [6, 7], "down_sampling": 1, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 8}, {"operations": [1, 1, "normal"], "inputs": [7, 8], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 9}, {"operations": [1, 1, "normal"], "inputs": [8, 9], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 10}, {"operations": [1, 1, "normal"], "inputs": [9, 10], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 11}, {"operations": [1, 1, "normal"], "inputs": [10, 11], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 12}, {"operations": [1, 1, "normal"], "inputs": [11, 12], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 13}, {"operations": [1, 1, "normal"], "inputs": [12, 13], "down_sampling": 0, "aux_weight": 0.4, "is_candidate": 0, "stop_gradient": 0, "id": 14}, {"operations": [1, 1, "reduction"], "inputs": [13, 14], "down_sampling": 1, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 15}, {"operations": [1, 1, "normal"], "inputs": [14, 15], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 16}, {"operations": [1, 1, "normal"], "inputs": [15, 16], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 17}, {"operations": [1, 1, "normal"], "inputs": [16, 17], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 18}, {"operations": [1, 1, "normal"], "inputs": [17, 18], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 19}, {"operations": [1, 1, "normal"], "inputs": [18, 19], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 20}, {"operations": [1, 1, "normal"], "inputs": [19, 20], "down_sampling": 0, "aux_weight": 1.0, "is_candidate": 0, "stop_gradient": 0, "id": 21}], "normal": [{"operations": [], "inputs": [], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 0}, {"operations": [], "inputs": [], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 1}, {"operations": [1, 9, 8, 22], "inputs": [1, 1, 1], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "extra_dict": {"ops_ids": [5, 4, 3], "fs_omega": [1.82760889755329e-10, -1.6420013959628221e-10, -1.161180110398341e-10]}, "stop_gradient": 0, "id": 3}, {"operations": [23, 1, 25, 22], "inputs": [1, 3, 1], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "extra_dict": {"ops_ids": [6, 5, 8], "fs_omega": [-5.326894828017181e-11, 4.851921295290218e-11, 3.683000432408434e-11]}, "stop_gradient": 0, "id": 4}, {"operations": [1, 1, 9, 22], "inputs": [1, 3, 1], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "extra_dict": {"ops_ids": [17, 11, 16], "fs_omega": [4.492043542692059e-11, 3.295496167621259e-11, 3.0641416487453554e-11]}, "stop_gradient": 0, "id": 5}, {"operations": [23, 1, 14, 14, 14, 11], "inputs": [1, 1, 3, 4, 5], "down_sampling": 0, "aux_weight": 0, "is_candidate": 0, "stop_gradient": 0, "id": 2}]}' \
--netmorph_method=hard \
--nr_gpu=4 \
--num_classes=1000 \
--num_init_use_all_gpu=2 \
--num_lstms=2 \
--output_type=int32 \
--prediction_feature=none \
--q_child_method=1 \
--q_hallu_method=1 \
--q_parent_method=1 \
--regularize_coef=const \
--regularize_const=0.0001 \
--s_type=conv3 \
--sgd_moment=0.9 \
--stem_channel_rate=3.2 \
--training_type=darts_imagenet \
--w_init=var_scale \

