#!/bin/bash

oat calibrate camera raw &
oat frameserve gige raw -c config.toml gige -i 0

