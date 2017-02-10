#!/bin/bash

case "$1" in

	run)
		oat view frame raw_b                                         &
		oat view pose p_b                                            &
        oat posisock std p_b -p                                      &
        oat buffer pose p p_b                                        &
        oat buffer frame raw raw_b                                   &
        oat posigen rand p -r 1000                                   &
		oat frameserve wcam raw
		;;

    calib)
		oat calibrate camera raw -s [9,6] -w 0.025                   &
		oat frameserve wcam raw
        ;;

	clean)
		oat clean raw p raw_b p_b
		;;

	*)
		echo $"Usage: $0 {run|calib|clean}"
		exit 1

esac

