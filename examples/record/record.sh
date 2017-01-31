#!/bin/bash

case "$1" in

	run)
        oat record -s raw -p p -n test -o                         &
		oat view pose p                                           &
		oat posidet diff mon p --tune                             &
		oat framefilt col raw mon -C GREY                         &
		oat frameserve wcam raw
		;;

    calib)

		oat calibrate camera raw -s [9,6] -w 0.025                &
		oat frameserve wcam raw
        ;;

	clean)

		oat clean raw mon p
		;;

	*)
		echo $"Usage: $0 {run|calib|clean}"
		exit 1

esac

