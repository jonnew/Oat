#!/bin/bash

# You need to have compiled Oat with Point Grey support and have two gige
# cameras installed on your computer to use this...

# Two cameras streaming with default parameters
oat frameserve gige raw0 -i 0 &
oat frameserve gige raw1 -i 1 &

# Look at both
oat view raw0 &
oat view raw1
