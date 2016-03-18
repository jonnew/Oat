oat framefilt mask raw flt -c test.toml framefilt-mask &
osleep 1
time oat frameserve test raw -f ./earth.jpg -c test.toml test
