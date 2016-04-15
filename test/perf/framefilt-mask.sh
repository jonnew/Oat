oat framefilt mask raw flt -c test.toml framefilt-mask &
osleep 1
time oat frameserve test raw -f $1 -c test.toml test
