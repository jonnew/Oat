#!/bin/bash

# Print the chessboard included with this example and paste onto a peice of cardboard
# Move the chessboard around under the camera in many orientations during detection
# Press the 'h' key while the calibration window is highlighted for more help.

oat calibrate camera raw -n maze -f ~/Desktop -h 10 -w 14 -W 0.017 &
oat frameserve wcam raw

