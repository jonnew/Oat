#!/bin/bash

case "$1" in

	run)
		oat view frame raw                                        &
		oat view pose p                                           &
		oat posidet aruco raw p -S [3,3] \
                                -l 0.03  \
                                -s 0.004 \
                                -c ./calibration.toml calibration \
                                --tune                            &
		oat frameserve wcam raw
		;;

    calib)
		oat calibrate camera raw -s [9,6] -w 0.025                &
		oat frameserve wcam raw
        ;;

	clean)
		oat clean raw p
		;;

	*)
		echo $"Usage: $0 {run|calib|clean}"
		exit 1

esac

