#oat record -i FINAL -n mouse -f ./				& #> log.txt &
#oat view FINAL 		 											& #> log.txt &
#oat decorate RAW FINAL -p POS -s -S 							& #> log.txt &
#oat posicom mean FGRN FBLU POS -c config.toml -k combine 		& #> log.txt &
oat posifilt kalman PBLU FBLU -c config.toml -k kalman  		& #> log.txt &
oat posifilt kalman PGRN FGRN -c config.toml -k kalman  		& #> log.txt &
oat posidet hsv SUB PGRN -c config.toml -k hsv_green  			& #> log.txt &
oat posidet hsv SUB PBLU -c config.toml -k hsv_blue  			& #> log.txt &
oat framefilt mask RAW SUB -c config.toml -k mask  				& #> log.txt &
oat frameserve file RAW -f mouse.mpg -c config.toml -k video 	  #> log.txt 
