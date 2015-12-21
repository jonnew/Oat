#!/bin/bash

case "$1" in

	run)
		oat view final                                        & 
		oat decorate raw final -p pr1 pr2 pr3 -SsR            & 
		oat posifilt region p3 pr3 -c config.toml region      & 
		oat posifilt region p2 pr2 -c config.toml region      & 
		oat posifilt region p1 pr1 -c config.toml region      & 
        oat posigen rand2D p3 -c config.toml p3               & 
        oat posigen rand2D p2 -c config.toml p2               & 
        oat posigen rand2D p1 -c config.toml p1               & 

		sleep 1
		oat frameserve file raw -f ~/Desktop/rat.avi -c config.toml video
		;;

	clean)

		oat clean raw p1 pr1 final
		;;

    kill)
        pkill -2 -f oat-*
        ;;

	*)
		echo $"Usage: $0 {run|clean|kill}"
		exit 1

esac

