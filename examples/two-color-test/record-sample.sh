#!/bin/bash

# Record video two test two color tracking on maze.
# JPN and AC
# 2015-06
# MWL@MIT
# Path to configuration files
CONFIG_PATH="./config.toml"

oat view raw &
sleep 0.1
oat record -i raw -f ./ -n two_color_test -d -F 30 &
sleep 0.1
oat frameserve gige raw -c "$CONFIG_PATH" -k gige
