#!/bin/bash         

#shutdown() {
#	# Get our process group id
#	PGID=$(ps -o pgid= $$ | grep -o [0-9]*)
#
#	# Kill it in a new new process group
#	setsid kill -- -$PGID
#	exit 0
#}
#
#trap "shutdown" SIGINT SIGTERM

terminator \
	--fullscreen \
	--title="Viewer" \
	-x sh -c "./bin/viewer final" &

sleep 1
terminator \
	--new-tab \
	-x sh -c "./bin/decorate filt vid final; bash" &

sleep 1
terminator \
	--new-tab \
	-x sh -c "./bin/posifilt kalman detect filt -c test_file_tracker_config.toml -k kalman; bash" &

sleep 1
terminator \
	--new-tab \
	-x sh -c "./bin/detector hsv vid detect -c test_file_tracker_config.toml -k hsv; bash" &
sleep 1
terminator \
	--new-tab \
 	-x sh -c "./bin/camserv file vid -c test_file_tracker_config.toml -k file_cam -f test_mouse.mpg; bash" &

