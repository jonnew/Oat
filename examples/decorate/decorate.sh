#!/bin/bash

case "$1" in

	6dof)
		oat view frame raw                                        &
		oat view pose p                                           &
		oat view frame dec                                        &
        oat decorate raw dec -p p -htsS -l 10 -f 0.75\
                             -c calibration.toml calibration      &
        oat posigen rand p -R [0,1,0,1,0,1] -u 1 -o               &
		oat frameserve wcam raw
		;;

	3dof)
		oat view frame dec                                        &
        oat decorate raw dec -p p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 \
                                p10 p11 p12 p13 p14 p15 p16 p17 \
                                p17 p19 \
                             -htsS -l 10 -f 1                     &
        oat posigen rand p19 -r 10 -R [0,640,0,480,0,0]           &
        oat posigen rand p18 -r 10 -R [0,640,0,480,0,0]           &
        oat posigen rand p17 -r 10 -R [0,640,0,480,0,0]           &
        oat posigen rand p16 -r 10 -R [0,640,0,480,0,0]           &
        oat posigen rand p15 -r 10 -R [0,640,0,480,0,0]           &
        oat posigen rand p14 -r 10 -R [0,640,0,480,0,0]           &
        oat posigen rand p13 -r 10 -R [0,640,0,480,0,0]           &
        oat posigen rand p12 -r 10 -R [0,640,0,480,0,0]           &
        oat posigen rand p11 -r 10 -R [0,640,0,480,0,0]           &
        oat posigen rand p10 -r 10 -R [0,640,0,480,0,0]           &
        oat posigen rand p9  -r 10 -R [0,640,0,480,0,0]           &
        oat posigen rand p8  -r 10 -R [0,640,0,480,0,0]           &
        oat posigen rand p7  -r 10 -R [0,640,0,480,0,0]           &
        oat posigen rand p6  -r 10 -R [0,640,0,480,0,0]           &
        oat posigen rand p5  -r 10 -R [0,640,0,480,0,0]           &
        oat posigen rand p4  -r 10 -R [0,640,0,480,0,0]           &
        oat posigen rand p3  -r 10 -R [0,640,0,480,0,0]           &
        oat posigen rand p2  -r 10 -R [0,640,0,480,0,0]           &
        oat posigen rand p1  -r 10 -R [0,640,0,480,0,0]           &
        oat posigen rand p0  -r 10 -R [0,640,0,480,0,0]           &
		oat frameserve wcam raw -r 10
		;;

    calib)

		oat calibrate camera raw -s [9,6] -w 0.025                &
		oat frameserve wcam raw
        ;;

	clean)

		oat clean raw p p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 \
                  p12 p13 p14 p15 p16 p17 p18 p19 dec
		;;

	*)
		echo $"Usage: $0 {6dof|3dof|calib|clean}"
		exit 1

esac
