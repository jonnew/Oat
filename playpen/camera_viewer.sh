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
gnome-terminal -x sh -c "./bin/viewer maze_cam; bash" &
wait
