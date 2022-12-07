#!/bin/bash
CACHE=$1
python train.py task=AllegroHandMAE headless=False pipeline=gpu \
task.env.numEnvs=1 test=True \
task.env.object.type=meat_can \
train.algo=PPO \
task.env.randomization.randomizeMass=False \
task.env.randomization.randomizeCOM=False \
task.env.randomization.randomizeFriction=False \
task.env.randomization.randomizePDGains=False \
task.env.randomization.randomizeScale=True \
train.ppo.priv_info=False \
train.ppo.output_name=AllegroHandMAE/"${CACHE}" \
checkpoint=outputs/AllegroHandMAE/"${CACHE}"/stage1_nn/best.pth

# task.env.object.type=simple_tennis_ball \
# task.env.object.type=banana \