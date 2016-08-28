oat decorate raw dec -p pos -tsSRh &
sleep 1
oat posigen rand2D pos -n 1000 &
time oat frameserve test raw -f $1 -c test.toml test

oat decorate raw dec -tsS &
sleep 1
time oat frameserve test raw -f $1 -c test.toml test
