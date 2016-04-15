#!/usr/bin/python

# 1. One frame source
# 2. Two position detectors
# 3. A position combiner
# 4. Posisock to shmem
# 5. Python script parses positions to ensure sample sync.

from subprocess import *

#pvi = Popen(["oat", "view", "filt"]).pid
pcm = Popen(["oat", "posicom", "mean", "pos1", "pos2", "pos"]).pid 
pd1 = Popen(["oat", "posidet", "hsv", "raw", "pos2", "-c", "config.toml", "hsv2"]).pid 
pd1 = Popen(["oat", "posidet", "hsv", "raw", "pos1", "-c", "config.toml", "hsv1"]).pid 
pfs = Popen(["oat", "frameserve", "test", "raw", "-f", "ada.jpg"]).pid

# /home/jon/public/Oat/oat/libexec/oat-frameserve", 
