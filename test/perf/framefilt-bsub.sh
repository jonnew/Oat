oat framefilt bsub raw flt &
sleep 1
time oat frameserve test raw -f $1 -c test.toml test
