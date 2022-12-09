#!/bin/bash
GPUS=$1
CACHE=$2
C=outputs/AllegroHandMAE/"${CACHE}"/stage1_nn/best.pth
CUDA_VISIBLE_DEVICES=${GPUS} \
python train_mae.py task=AllegroHandMAE headless=True \
task.env.numEnvs=20000 test=True task.on_evaluation=True \
task.env.object.type=meat_can \
train.algo=PPO \
task.env.randomization.randomizeMass=True \
task.env.randomization.randomizeCOM=True \
task.env.randomization.randomizeFriction=True \
task.env.randomization.randomizePDGains=True \
task.env.randomization.randomizeScale=True \
task.env.randomization.jointNoiseScale=0.005 \
task.env.reset_height_threshold=0.6 \
task.env.forceScale=2 task.env.randomForceProbScalar=0.25 \
train.ppo.priv_info=False train.ppo.proprio_adapt=False \
train.ppo.output_name=AllegroHandHora/"${CACHE}" \
checkpoint="${C}"

# task.env.object.type=cylinder_default \