#!/bin/bash

case "$1" in

	run)

		#oat record -i FINAL -n mouse -f ./                         2>&1 | tee log.txt &
		oat view FINAL 		 								        2>&1 | tee log.txt &
		oat decorate RAW FINAL -p POS -sS 					        2>&1 | tee log.txt &
		oat posicom mean FGRN FBLU POS -c config.toml combine       2>&1 | tee log.txt &
		oat posifilt kalman PBLU FBLU -c config.toml kalman         2>&1 | tee log.txt &
		oat posifilt kalman PGRN FGRN -c config.toml kalman         2>&1 | tee log.txt &
		oat posidet hsv SUB PGRN -c config.toml hsv_green  	        2>&1 | tee log.txt &
		oat posidet hsv SUB PBLU -c config.toml hsv_blue  	        2>&1 | tee log.txt &
		oat framefilt mask RAW SUB -c config.toml mask  	        2>&1 | tee log.txt &

		sleep 1
		oat frameserve file RAW -f mouse.mpg -c config.toml video   2>&1 | tee log.txt 
		;;

	clean)

		oat clean RAW SUB PBLU PGRN FGRN FBLU POS FINAL
		;;

	*)
		echo $"Usage: $0 {run|clean}"
		exit 1

esac
