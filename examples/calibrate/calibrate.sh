#!/bin/bash          

oat frameserve gige raw &

oat calibrate camera raw -c config.toml -k gige
