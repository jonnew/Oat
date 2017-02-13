#!/bin/bash

# Two test two color tracking

case "$1" in

	run)

        ## view data, & means runs in background
        #oat view RAW &
        #
        ## look at mask overlaid on video to make sure it's ok
        #oat view FINAL &
        #
        #oat framefilt bsub RAW SUB &

        # decorate
        sleep 0.1
        oat decorate RAW FINAL -p ORNG BLUE -sSRt &

        sleep 0.1
        oat posifilt kalman COMBO FILT -c config.toml -k kalman &

        sleep 0.1
        oat posicom mean BLUE ORNG COMBO -c config.toml -k mean &

        # detecting orange and blue leds in raw data
        sleep 0.1
        oat posidet hsv SUB ORNG -c config.toml -k hsv_orange &
        oat posidet hsv SUB BLUE -c config.toml -k hsv_blue &

        # apply mask to determine area of interest, path to mask file is in config
        sleep 0.1
        oat framefilt mog AOI SUB &

        # apply mask to determine area of interest, path to mask file is in config
        sleep 0.1
        oat framefilt mask RAW AOI -c config.toml -k mask &

        # read file, get data, use settings under [video] in config file (sets frame rate)
        sleep 0.1
        oat frameserve file RAW -f ./two_color_test_raw.avi -c config.toml -k video
        ;;

	clean)

		oat clean RAW AOI SUB ORNG BLUE COMBO FILT FINAL
		;;

	*)
		echo $"Usage: $0 {run|clean}"
		exit 1

esac
