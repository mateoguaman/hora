#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python train_mae.py task=AllegroHandMAE headless=True seed=${SEED} \
task.env.forceScale=2 task.env.randomForceProbScalar=0.25 \
train.algo=PPO \
task.env.object.type=cylinder_default \
train.ppo.priv_info=False train.ppo.proprio_adapt=False \
train.ppo.output_name=AllegroHandMAE/"${CACHE}" \
train.load_path=/home/mihirpd_google_com/projects/mateo_hora/hora/outputs/AllegroHandMAE/mae/stage1_nn/ep_9500_step_1245M_reward_42.33.pth \
${EXTRA_ARGS}