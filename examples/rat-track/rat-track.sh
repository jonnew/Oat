#!/bin/bash

case "$1" in

	run)
		#oat record -i final -f ./ -n result -F 30 &
		#sleep 0.1
		#oat record -p pix wld -f ./ -n result &
		sleep 0.1
		oat posifilt homography posi wld -c config.toml -k homography &
		sleep 0.1
		oat view final &
		sleep 0.1
		oat decorate raw final -p pix -SsR &
		sleep 0.1
		oat posifilt region posi pix -c config.toml -k region &
		sleep 0.1
		oat posifilt kalman det posi -c config.toml -k kalman &
		sleep 0.1
		oat posidet hsv bac det -c config.toml -k hsv &
		sleep 0.1
		oat framefilt mog raw bac &
		#sleep 0.1
		#oat framefilt mask raw roi -c config.toml -k mask &
		sleep 0.1
		oat frameserve wcam raw #file raw -f ./rat.avi -c config.toml -k video  
		;;

	clean)

		oat clean raw roi bac det posi final pix wld
		;;

	*)
		echo $"Usage: $0 {run|clean}"
		exit 1

esac

