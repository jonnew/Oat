#!/bin/bash

case "$1" in

	run)
        oat record --p p -do -n "test" -f ./        & # Copy of pose in JSON format
        oat record -s raw -p p -dbo -n "test" -f ./ & # Uncompressed video and binary pose
		oat posidet diff mon p --tune               &
		oat framefilt col raw mon -C GREY           &
		oat frameserve wcam raw
		;;

	clean)

		oat clean raw mon p
		;;

	*)
		echo $"Usage: $0 {run|calib|clean}"
		exit 1

esac

