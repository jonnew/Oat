#!/bin/bash          

oat calibrate camera raw &
sleep 0.1
oat frameserve gige raw -c config.toml gige
