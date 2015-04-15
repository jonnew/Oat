#!/bin/bash         

shutdown() {
	# Get our process group id
	PGID=$(ps -o pgid= $$ | grep -o [0-9]*)

	# Kill it in a new new process group
	setsid kill -- -$PGID
	exit 0
}

trap "shutdown" SIGINT SIGTERM

# Start a bunch of child processes that are killable with CTRL-C
gnome-terminal -x sh -c "./bin/camserv gige maze_cam -c config.toml -k maze_cam; bash" &
sleep 3
gnome-terminal -x sh -c "./bin/backsub maze_cam back_out; bash" &
sleep 1
gnome-terminal -x sh -c "./bin/hsvdetector back_out post --framesink blue_det -c config.toml -k blue_hsv; bash" &
gnome-terminal -x sh -c "./bin/hsvdetector back_out ant --framesink orange_det -c config.toml -k orange_hsv; bash" &
sleep 1
gnome-terminal -x sh -c "./bin/viewer maze_cam; bash" &
gnome-terminal -x sh -c "./bin/viewer back_out; bash" &
gnome-terminal -x sh -c "./bin/viewer blue_det; bash" &
gnome-terminal -x sh -c "./bin/viewer orange_det; bash" &

wait

