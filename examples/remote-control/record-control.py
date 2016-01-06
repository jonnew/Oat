#!/usr/bin/python

"""
A stupid GUI to control oat-record remotely

Warning: I don't know how to write Python.
"""
import collections
import Tkinter as tk
import tkFont
import zmq

CTX = zmq.Context()

# Device tuple
Device = collections.namedtuple('Device', ['name', 
                                           'socket', 
                                           'start_cmd', 
                                           'stop_cmd', 
                                           'newfile_cmd', 
                                           'exit_cmd'])

# Hard-coded devices along with appropriate commands. Add more or remove the ones you don't want
DEVICES = [
    Device("Open Ephys", CTX.socket(zmq.PUB), "StartRecord", "StopRecord", "NewFile",""),
    Device("Oat", CTX.socket(zmq.PUB), "start\n", "stop\n", "new\n", "exit\n"),
    Device("Maze", CTX.socket(zmq.PUB), "start", "stop", "new", "exit\n")
]

# Generic remote connction specification
class RemoteConnection(tk.Frame):

    def __init__(self, parent, device):

        tk.Frame.__init__(self, parent)

        self.parent = parent
        self.font = parent.font
        self.device = device
        self.connected = False
        self.addr = None
        self.conn_addr = None

        self.initUI()

    def initUI(self):

        # Grid config
        self.columnconfigure(1, weight=1)

        # Label
        label = tk.Label(self, font=self.font, text=self.device.name, width=10, anchor=tk.W)
        label.grid(row=0, column=0, padx=10, sticky=tk.W)

        # Text entry
        entry = tk.Entry(self, font=self.font)
        entry.grid(row=0, column=1, sticky=tk.W+tk.E)
        entry.bind('<Leave>', lambda event: self.updateEndpoint(event))

        # Connect button
        b_conn_txt = tk.StringVar()
        b_conn_txt.set("Connect")
        b_conn = tk.Button(self, textvariable=b_conn_txt, font=self.font,
                command = lambda: self.connect(b_conn_txt, label))
        b_conn.grid(row=0, column=2, padx=10, sticky=tk.E)

    # Connect/Disconnect from remote endpoint
    def connect(self, txt, label):
        if not self.connected:
            if self.addr is not None:
                try:
                    self.device.socket.bind(self.addr)
                except zmq.ZMQError:
                    self.addr = None
                    print ("Failed: Invalid " + self.device.name + " endpoint.")
                    return

                self.conn_addr = self.addr
                self.connected = True
                label.config(fg='green')
                txt.set("Disconnect")
        else:
            try:
                self.device.socket.unbind(self.conn_addr)
            except zmq.ZMQError:
                print ("Failed to disconnected from " + self.device.name + " endpoint.")

            self.connected = False
            label.config(fg='black')
            txt.set("Connect")

    # Udpate socket address
    def updateEndpoint(self, event):

        txt = event.widget.get()
        if txt:
            self.addr = txt
        else:
            self.addr = None


# Basic GUI
class RemoteControl(tk.Frame):

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)

        self.parent = parent
        self.font = parent.font

        # The connections we are interested in
        self.connections = [RemoteConnection(self, dev) for dev in DEVICES]

        self.initUI()

    def initUI(self):

        self.pack(fill=tk.BOTH)

        # Connection UIs
        i = 0
        for i,c in enumerate(self.connections):
            c.grid(row=i, column=0, columnspan=3, pady=5)
        i+=1

        # Record control buttons
        b_start = tk.Button(self, text="Start", font=self.font, command=self.startRecording)
        b_stop = tk.Button(self, text="Stop", font=self.font, command=self.stopRecording)
        b_new = tk.Button(self, text="New", font=self.font, command=self.makeNewFile)
        b_exit = tk.Button(self, text="Exit", font=self.font, command=self.exitAll)

        b_start.grid(row=i, column=0, padx=10, pady=10, sticky=tk.W)
        b_stop.grid(row=i, column=1, padx=10, pady=10, sticky=tk.W)
        b_new.grid(row=i, column=2, padx=10, pady=10, sticky=tk.W)
        b_exit.grid(row=i, column=4, padx=10, pady=10, sticky=tk.W)

    def startRecording(self):
        for i, conn in enumerate(self.connections):
            if conn.connected:
                conn.device.socket.send(conn.device.start_cmd)

    def stopRecording(self):
        for i, conn in enumerate(self.connections):
            if conn.connected:
                conn.device.socket.send(conn.device.stop_cmd)

    def makeNewFile(self):
        print ("New file")

    def exitAll(self):
        for i, conn in enumerate(self.connections):
            if conn.connected:
                conn.device.socket.send(conn.device.exit_cmd)

def main():

    root = tk.Tk()
    root.title("Stupid Controller")
    root.font = tkFont.Font(family="Helvetica", size=12)
    root.geometry("250x150+300+300")
    app = RemoteControl(root)
    root.mainloop()

if __name__ == '__main__':
    main()
