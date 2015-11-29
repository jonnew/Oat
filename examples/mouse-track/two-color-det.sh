#!/bin/bash

case "$1" in

	run)

		#oat record -i FINAL -n mouse -f ./                         & 
		oat view FINAL 		 								        & 
		oat decorate RAW FINAL -p POS -sS 					        & 
		oat posicom mean FGRN FBLU POS -c config.toml combine       & 
		oat posifilt kalman PBLU FBLU -c config.toml kalman         & 
		oat posifilt kalman PGRN FGRN -c config.toml kalman         & 
		oat posidet hsv SUB PGRN -c config.toml hsv_green  	        & 
		oat posidet hsv SUB PBLU -c config.toml hsv_blue  	        & 
		oat framefilt mog RAW SUB   	                            & 

		sleep 1
		oat frameserve file RAW -f mouse.mpg -c config.toml video   &
		;;

	clean)

		oat clean RAW SUB PBLU PGRN FGRN FBLU POS FINAL
		;;

	*)
		echo $"Usage: $0 {run|clean}"
		exit 1

esac
