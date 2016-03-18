oat framefilt undistort raw flt -c test.toml framefilt-undistort &
sleep 1
time oat frameserve test raw -f ./earth.jpg -c test.toml test
