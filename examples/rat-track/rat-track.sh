#!/bin/bash

case "$1" in

	run)
		#oat record -i final -f ./ -n result -F 30                  &
		#oat record -p pix wld -f ./ -n result                      &
		oat posifilt homography posi wld -c config.toml homography  &
		oat view final                                              &
		oat decorate raw final -p pix -SsR                          &
		oat posifilt region posi pix -c config.toml region          &
		oat posifilt kalman det posi -c config.toml kalman          &
		oat posidet hsv bac det -c config.toml hsv                  &
		oat framefilt mog raw bac                                   &
		#oat framefilt mask raw roi -c config.toml mask             &

		sleep 1
		oat frameserve wcam raw #file raw -f ./rat.avi -c config.toml video  
		;;

	clean)

		oat clean raw roi bac det posi final pix wld
		;;

	*)
		echo $"Usage: $0 {run|clean}"
		exit 1

esac

