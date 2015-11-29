#!/bin/bash

# Obviously, put whatever IP and Port in there that you want to use

case "$1" in

	run)

        oat posisock udp pos -h 10.121.43.222 -p 5555       &

		sleep 1
        oat positest rand2D pos -r 100
		;;

	clean)

		oat clean pos
		;;

	*)
		echo $"Usage: $0 {run|clean}"
		exit 1

esac
