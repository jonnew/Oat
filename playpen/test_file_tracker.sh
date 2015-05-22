#!/bin/bash         

terminator \
	--fullscreen \
	--title="Viewer" \
	-x sh -c "./bin/viewer final" &

sleep 1
terminator \
	--new-tab \
	-x sh -c "./bin/record -i final -f ~/Desktop -d; bash" &

sleep 1
terminator \
	--new-tab \
	-x sh -c "./bin/decorate -p filt vid final; bash" &

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
 	-x sh -c "./bin/frameserve file vid -c test_file_tracker_config.toml -k file_cam -f test_mouse.mpg; bash" &

