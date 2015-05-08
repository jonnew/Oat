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
gnome-terminal -x sh -c "./bin/viewer final_image; bash" &
sleep 1
gnome-terminal -x sh -c "./bin/decorate position maze_cam final_image; bash" &
sleep 1
gnome-terminal -x sh -c "./bin/detector diff maze_cam ant -c config.toml -k orange_hsv; bash" &
sleep 1
gnome-terminal -x sh -c "./bin/camserv wcam maze_cam; bash" &

wait

