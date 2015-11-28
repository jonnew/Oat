#!/bin/bash

case "$1" in

	run)

		#oat record -i FINAL -n mouse -f ./				                & #> log.txt &
		sleep 0.1
		oat view FINAL 		 											& #> log.txt &
		sleep 0.1
		oat decorate RAW FINAL -p POS -sS 							    & #> log.txt &
		sleep 0.1
		oat posicom mean FGRN FBLU POS -c config.toml -k combine 		& #> log.txt &
		sleep 0.1
		oat posifilt kalman PBLU FBLU -c config.toml -k kalman  		& #> log.txt &
		sleep 0.1
		oat posifilt kalman PGRN FGRN -c config.toml -k kalman  		& #> log.txt &
		sleep 0.1
		oat posidet hsv SUB PGRN -c config.toml -k hsv_green  			& #> log.txt &
		sleep 0.1
		oat posidet hsv SUB PBLU -c config.toml -k hsv_blue  			& #> log.txt &
		sleep 0.1
		oat framefilt mask RAW SUB -c config.toml -k mask  				& #> log.txt &
		sleep 0.1
		oat frameserve file RAW -f mouse.mpg -c config.toml -k video 	  #> log.txt 
		;;

	clean)

		oat clean RAW SUB PBLU PGRN FGRN FBLU POS FINAL
		;;

	*)
		echo $"Usage: $0 {run|clean}"
		exit 1

esac
