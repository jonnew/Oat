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
gnome-terminal -x sh -c "./bin/viewer final1; bash" &
sleep 1
gnome-terminal -x sh -c "./bin/decorate hsv1 vid1 final1; bash" &
sleep 1
#gnome-terminal -x sh -c "./bin/posifilt kalman hsv filt -c test_file_tracker_config.toml -k kalman; bash" &
sleep 1
gnome-terminal -x sh -c "./bin/detector hsv vid1 hsv1 -c test_file_tracker_config.toml -k hsv; bash" &
sleep 1
gnome-terminal -x sh -c "./bin/camserv file vid1 -c test_file_tracker_config.toml -k file_cam -f test_mouse.mpg; bash" &
wait

