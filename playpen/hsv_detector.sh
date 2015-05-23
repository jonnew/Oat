#!/bin/bash
SESSION=$USER

tmux -2 new-session -d -s $SESSION

# Create a new window to house this configuration
tmux new-window -t $SESSION:1 -n 'Basic Viewer'
tmux split-window -h
tmux split-window -h
tmux split-window -h
tmux split-window -h
tmux split-window -h
tmux select-layout tiled

# Pane 1: viewer
tmux select-pane -t 5
#tmux send-keys "./record -i final -f ~/Desktop -d" C-m

# Pane 1: viewer
tmux select-pane -t 4
tmux send-keys "./viewer final" C-m

# Pane 1: viewer
tmux select-pane -t 3
tmux send-keys "./decorate -p filt raw final" C-m

# Pane 1: viewer
tmux select-pane -t 2
tmux send-keys "./posifilt kalman det filt" C-m

# Pane 1: viewer
tmux select-pane -t 1
tmux send-keys "./detector hsv raw det" C-m

# Pane 0: frameserve
tmux select-pane -t 0
tmux send-keys "./frameserve gige raw" C-m
tmux set-window-option synchronize-panes on

tmux -2 attach-session -t $SESSION
