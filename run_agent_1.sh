#!/bin/bash

export CARLA_ROOT=../CARLA_0.9.10.1           # change to where you installed CARLA
export PORT=2500                                                    # change to port that CARLA is running on
export ROUTES=./leaderboard/data/routes_training/route_12.xml         # change to desired route
export TEAM_AGENT=shawan_agent.py                                    # no need to change
export TEAM_CONFIG=./checkpoints/epoch=24.ckpt                                     # change path to checkpoint
export HAS_DISPLAY=1                                                # set to 0 if you don't want a debug window
export REPS=10000
export TRAFFIC=2700
export ID=0


./run_agent.sh
