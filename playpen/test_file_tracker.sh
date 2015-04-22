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
#gnome-terminal -x sh -c "./bin/viewer f11; bash" &
#sleep 1
#gnome-terminal -x sh -c "./bin/decorate h11 v11 f11; bash" &
#sleep 1
#gnome-terminal -x sh -c "./bin/detector hsv v11 h11 -c test_file_tracker_config.toml -k blue_hsv; bash" &
#sleep 1
gnome-terminal -x sh -c "./bin/camserv file v11 -c test_file_tracker_config.toml -k file_cam -f test_mouse.mpg; bash" &
wait

