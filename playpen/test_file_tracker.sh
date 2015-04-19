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
gnome-terminal -x sh -c "./bin/camserv file maze_cam -c test_file_tracker_config.toml -k file_cam -f test_mouse.mpg; bash" &
sleep 3
gnome-terminal -x sh -c "./bin/detector hsv maze_cam hsv_out -c test_file_tracker_config.toml -k blue_hsv; bash" &
sleep 1
gnome-terminal -x sh -c "./bin/decorate hsv_out maze_cam final_image; bash" &
sleep 1
gnome-terminal -x sh -c "./bin/viewer final_image; bash" &
wait

