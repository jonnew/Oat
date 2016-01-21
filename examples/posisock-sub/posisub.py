#!/bin/python

# Example python script that will listen to oat posisock pub -e "tcp://*:5555"
# and print received positions to command line

import sys
import zmq

#  Socket to talk to server
context = zmq.Context()
socket = context.socket(zmq.SUB)

print("Listening for position updates on tcp://localhost:5555")
socket.connect("tcp://localhost:5555")

# Subscribe to all
pos_filter = ""

# Python 2 - ascii bytes to unicode str
if isinstance(pos_filter, bytes):
    pos_filter = pos_filter.decode('ascii')
socket.setsockopt_string(zmq.SUBSCRIBE, pos_filter)

# Process 5 updates
total_temp = 0
while True:
    string = socket.recv_string()
    print string
