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
./bin/camserv_ge maze_cam config.toml maze_cam				&
sleep 3
./bin/backsubtractor maze_cam back_out 						& 
sleep 1
./bin/hsvdetector back_out blue_det config.toml blue_hsv	&
./bin/hsvdetector back_out orange_det config.toml orange_hsv	&
sleep 1
./bin/viewer maze_cam 						&
./bin/viewer back_out 						&
./bin/viewer blue_det 						&
./bin/viewer orange_det 						&

wait

