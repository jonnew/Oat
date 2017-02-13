#!/bin/bash

case "$1" in

	run)
		oat view final                                        &
		oat decorate raw final -p pr1 pr2 pr3 pcr -SsR        &
		oat posifilt region pc pcr -c config.toml region      &
		oat posifilt region p3 pr3 -c config.toml region      &
		oat posifilt region p2 pr2 -c config.toml region      &
		oat posifilt region p1 pr1 -c config.toml region      &
        oat posicom mean p1 p2 pc -c config.toml combiner     &
        oat posigen rand2D p3 -c config.toml pgen             &
        oat posigen rand2D p2 -c config.toml pgen             &
        oat posigen rand2D p1 -c config.toml pgen             &

		sleep 1
		oat frameserve file raw -f ~/Desktop/rat.avi -c config.toml video
		;;

	clean)

		oat clean raw p1 pr1 final
		;;

	*)
		echo $"Usage: $0 {run|clean}"
		exit 1

esac

