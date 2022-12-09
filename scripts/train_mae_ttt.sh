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
python train_mae.py task=AllegroHandMaeTTT headless=True seed=${SEED} \
task.env.forceScale=2 task.env.randomForceProbScalar=0.25 \
train.algo=PPO \
task.env.object.type=cylinder_default \
train.ppo.priv_info=False train.ppo.proprio_adapt=False \
train.ppo.output_name=AllegroHandMaeTTT/"${CACHE}" \
train.load_path=/home/mihirpd_google_com/projects/mateo_hora/hora/outputs/AllegroHandMAE/mae/stage1_nn/best.pth \
${EXTRA_ARGS}