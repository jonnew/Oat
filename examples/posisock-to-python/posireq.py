#!/bin/python

# Example python script that will synchronously request positions
# from posisock rep -e "tcp://*:5555" and print received positions to
# command line

import sys
import zmq

#  Socket to talk to server
context = zmq.Context()
socket = context.socket(zmq.REQ)

print("Requesting position updates from tcp://localhost:5555")
socket.connect("tcp://localhost:5555")

# Request positions forever
total_temp = 0
while True:
    socket.send(b"gimme") # Can be anything, currently.
    position = socket.recv_string()
    print(position)

