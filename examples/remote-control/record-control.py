#!/usr/bin/python

"""
A stupid GUI to control oat-record and some other crap remotely.

Warning: I don't know how to write Python.
"""

try:
    import tkinter as tk
    import tkinter.font as tkf
except:
    import Tkinter as tk
    import tkFont as tkf

import collections
import zmq
import signal
import sys

# Exit routine
def signal_handler(signal, frame):
    print('Exiting')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# RPC command set
RPCInterface = collections.namedtuple('RPC', ['socket', 'poll'])

class RemoteFrontPanel(object):

    def __init__(self, name, addr, help_cmd="help", start_cmd="start",
            pause_cmd="pause", newfile_cmd="new", quit_cmd="quit"):

        self.ctx = zmq.Context()
        self.name = name
        self.is_connected = False
        self.help_cmd = help_cmd
        self.start_cmd = start_cmd
        self.pause_cmd = pause_cmd
        self.newfile_cmd = newfile_cmd
        self.quit_cmd = quit_cmd
        self.req_addr = addr
        self.conn_addr = None
        self.retries = 3
        self.retries_left = self.retries
        self.request_timeout = 100 # msec

    def __enter__(self):
        return self

    def connect(self):
        self.rpc = RPCInterface(self.ctx.socket(zmq.REQ), zmq.Poller())
        self.rpc.socket.connect(self.req_addr)
        self.rpc.poll.register(self.rpc.socket, zmq.POLLIN)
        self.is_connected = True

    def disconnect(self):
        self.rpc.socket.setsockopt(zmq.LINGER, 0)
        self.rpc.socket.close()
        self.rpc.poll.unregister(self.rpc.socket)
        self.is_connected = False

    def sendMsg(self, request):
        
        print("I [%s]: Sending \'%s\'\n" % (self.name, request.rstrip()))
        self.rpc.socket.send_string(str(request))

        expect_reply = True
        while expect_reply:
            socks = dict(self.rpc.poll.poll(self.request_timeout))
            if socks.get(self.rpc.socket) == zmq.POLLIN:
                reply = self.rpc.socket.recv()
                if not reply:
                    break
                else:
                    print("I [%s]: Received: %s" % (self.name, reply))
                    self.retries_left = self.retries
                    expect_reply = False
            else:
                print("W [%s]: No response from server, retrying...", self.name)
                # REQ/REP socket is in confused state. Close it and create another.
                self.disconnect()
                self.retries_left -= 1
                self.connect()

                if self.retries_left == 0:
                    self.retries_left = self.retries
                    print("E: [%s] Server offline, abondoning." % self.name)
                    break

                # If we have not exhausted retries, try again
                print("I [%s]: Reconnecting and resending (%s)" % (self.name, request))
                self.rpc.socket.send_string(str(request))

    def getHelp(self):
        if self.help_cmd:
            self.sendMsg(self.help_cmd)

    def sendStart(self):
        if self.start_cmd:
            self.sendMsg(self.start_cmd)

    def sendStop(self):
        if self.pause_cmd:
            self.sendMsg(self.pause_cmd)

    def makeNewFile(self):
        if self.newfile_cmd:
            self.sendMsg(self.newfile_cmd)

    def sendQuit(self):
        if self.quit_cmd:
            self.sendMsg(self.quit_cmd)
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.is_connected:
            self.disconnect()
        self.ctx.term()

# Common file name
class FileName(tk.Frame):

    def __init__(self, parent):

        tk.Frame.__init__(self, parent)

        self.parent = parent
        self.font = parent.font

        self.filename = None

        self.initUI()

    def initUI(self):

        # Grid config
        self.columnconfigure(1, weight=1)

        # Label
        label = tk.Label(self, font=self.font, text="File name", width=15, anchor=tk.W)
        label.grid(row=0, column=0, padx=10, sticky=tk.W)

        # Text entry
        entry = tk.Entry(self, font=self.font)
        entry.delete(0, tk.END)
        entry.insert(0, "file")
        entry.grid(row=0, column=1, sticky=tk.W+tk.E)
        entry.bind('<Leave>', lambda event: self.updateFileName(event))

    # Udpate socket address
    def updateFileName(self, event):

        txt = event.widget.get()
        if txt:
            self.filename = txt
        else:
            self.filename = None

# Generic remote connction for interacting with a single device
class RemoteConnection(tk.Frame):

    def __init__(self, parent, device):

        tk.Frame.__init__(self, parent)

        self.parent = parent
        self.font = parent.font

        # RemoteFrontPanel reference
        self.device = device

        self.initUI()

    def initUI(self):

        # Grid config
        self.columnconfigure(1, weight=1)

        # Label
        label = tk.Label(self, font=self.font, text=self.device.name, width=15, anchor=tk.W)
        label.grid(row=0, column=0, padx=10, sticky=tk.W)

        # Text entry
        entry = tk.Entry(self, font=self.font)
        entry.delete(0, tk.END)
        entry.insert(0, self.device.req_addr)
        entry.grid(row=0, column=1, sticky=tk.W+tk.E)
        entry.bind('<Leave>', lambda event: self.updateEndpoint(event))

        # Connect button
        b_conn_txt = tk.StringVar()
        b_conn_txt.set("Connect")
        b_conn = tk.Button(self, textvariable=b_conn_txt, font=self.font, width=15,
                command = lambda: self.connect(b_conn_txt, label))
        b_conn.grid(row=0, column=2, padx=10, sticky=tk.E)

    # Connect/Disconnect from remote endpoint
    def connect(self, txt, label):
        if not self.device.is_connected:
            try:
                self.device.connect()
            except zmq.ZMQError:
                self.device.conn_addr = None
                print ("Failed: Invalid " + self.device.name + " endpoint.")
                return

            self.device.conn_addr = self.device.req_addr
            label.config(fg='green')
            txt.set("Disconnect")
        else:
            try:
                self.device.disconnect()
            except zmq.ZMQError:
                print ("Failed to disconnected from " + self.device.name + " endpoint.")

            label.config(fg='black')
            txt.set("Connect")

    # Udpate socket address
    def updateEndpoint(self, event):

        txt = event.widget.get()
        if txt:
            self.device.req_addr = txt
        else:
            self.device.req_addr = None

# Basic GUI
class RemoteControl(tk.Frame):

    def __init__(self, parent, devices):
        tk.Frame.__init__(self, parent)

        self.parent = parent
        self.font = parent.font

        # Common file name
        self.filename = FileName(self)

        # The devices we are interested in contolling
        self.connections = [RemoteConnection(self, dev) for dev in devices]

        self.initUI()

    def initUI(self):

        self.config(borderwidth=2, relief=tk.RAISED)
        self.pack()

        # Current row counter
        i = 0

        # Common file name to issue during call to 'new'. Empty means use the
        # same one that the remot recorder already was using.
        l_filename = tk.Label(self, text="Common File Name", font=self.font)
        l_filename.grid(row=i, column=0)
        i+=1

        self.filename.grid(row=i, column=0, pady=5, sticky="ew")
        i+=1

        # Connection UIs
        l_endpoints = tk.Label(self, text="Remote Endpoints", font=self.font)
        l_endpoints.grid(row=i, column=0)
        i+=1

        # Align the Remote Connection components
        start_row = i
        for j,c in enumerate(self.connections):
            i = start_row + j
            c.grid(row=i, column=0, pady=5, sticky="ew")
        i += 1

        b_frame = tk.Frame(self)

        # Record control buttons
        l_controls = tk.Label(self, text="Remote Controls", font=self.font)
        l_controls.grid(row=i, column=0)
        i+=1

        b_help = tk.Button(b_frame, text="Help", font=self.font, command=self.printHelp)
        b_start = tk.Button(b_frame, text="Start", font=self.font, command=self.startRecording)
        b_stop = tk.Button(b_frame, text="Stop", font=self.font, command=self.stopRecording)
        b_new = tk.Button(b_frame, text="New", font=self.font, command=self.makeNewFile)
        b_exit = tk.Button(b_frame, text="Exit", font=self.font, command=self.quitAll)

        b_help.pack(side="left", fill=None, expand=False, padx=10)
        b_start.pack(side="left", fill=None, expand=False, padx=10)
        b_stop.pack(side="left", fill=None, expand=False, padx=10)
        b_new.pack(side="left", fill=None, expand=False,  padx=10)
        b_exit.pack(side="left", fill=None, expand=False, padx=10)

        b_frame.grid(row=i, column=0, sticky="w", padx=10)

    def printHelp(self):
        for i, conn in enumerate(self.connections):
            if conn.device.is_connected:
                conn.device.getHelp()

    def startRecording(self):
        for i, conn in enumerate(self.connections):
            if conn.device.is_connected:
                conn.device.sendStart()

    def stopRecording(self):
        for i, conn in enumerate(self.connections):
            if conn.device.is_connected:
                conn.device.sendStop()

    def makeNewFile(self):
        for i, conn in enumerate(self.connections):
            if conn.device.is_connected:
                conn.device.makeNewFile()

    def quitAll(self):
        for i, conn in enumerate(self.connections):
            if conn.device.is_connected:
                conn.device.sendQuit()

def main():

    devices = [ RemoteFrontPanel("Open Ephys", "tcp://localhost:5556", "", "StartRecord", "StopRecord", "NewFile",""),
                RemoteFrontPanel("Oat", "tcp://localhost:6666", "help\n", "start\n", "pause\n", "new\n", "quit\n"),
                RemoteFrontPanel("Maze", "tcp://localhost:6665", "help", "start", "pause", "new", "exit")
              ] 

    root = tk.Tk()
    root.title("Stupid Controller")
    root.font = tkf.Font(family="Helvetica", size=12)
    app = RemoteControl(root, devices)
    root.mainloop()

if __name__ == '__main__':
    main()
