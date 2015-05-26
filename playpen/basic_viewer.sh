#!/bin/bash
SESSION=$USER

tmux -2 new-session -d -s $SESSION

# Create a new window to house this configuration
tmux rename-window 'Basic Viewer'
tmux split-window -h

# Pane 1: viewer
tmux select-pane -t 1
tmux send-keys "./viewer out" C-m

# Pane 0: frameserve
tmux select-pane -t 0
tmux send-keys "./frameserve gige out -c basic_viewer_config.toml -k maze_cam" C-m
tmux set-window-option synchronize-panes on

tmux -2 attach-session -t $SESSION
