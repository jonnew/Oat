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

RPCInterface = collections.namedtuple('RPC', ['socket',
                                              'help_cmd',
                                              'start_cmd',
                                              'stop_cmd',
                                              'newfile_cmd',
                                              'exit_cmd'])
# Device tuple
class Device(object):

    def __init__(self, name, addr, help_cmd, start_cmd, stop_cmd, newfile_cmd, exit_cmd):

        self.name = name
        self.is_connected = False
        self.rpc = RPCInterface(CTX.socket(zmq.REQ), help_cmd, start_cmd, stop_cmd, newfile_cmd, exit_cmd)
        self.req_addr = addr
        self.conn_addr = None

    def connect(self):
        self.rpc.socket.connect(self.req_addr)

    def disconnect(self):
        self.rpc.socket.disconnect(self.conn_addr)

    def sendMsg(self, request):
        self.rpc.socket.send(request)
        reply = self.rpc.socket.recv()
        print("[%s] Sent: \'%s\' and received\n" % (self.name, request.rstrip()))
        print("%s" % (reply))

    def getHelp(self):
        self.sendMsg(self.rpc.help_cmd)

    def sendStart(self):
        self.sendMsg(self.rpc.start_cmd)

    def sendStop(self):
        self.sendMsg(self.rpc.stop_cmd)

    def makeNewFile(self):
        self.sendMsg(self.rpc.newfile_cmd)

    def exit(self):
        self.sendMsg(self.rpc.exit_cmd)

# Hard-coded devices along with appropriate commands. Add more or remove the ones you don't want
DEVICES = [
    Device("Open Ephys", "tcp://localhost:5556", "", "StartRecord", "StopRecord", "NewFile",""),
    Device("Oat", "tcp://localhost:5557", "help\n", "start\n", "pause\n", "new\n", "exit\n"),
    Device("Maze", "tcp://localhost:5558", "help", "pause", "stop", "new", "exit")
]


# Generic remote connction for interacting with a single device
class RemoteConnection(tk.Frame):

    def __init__(self, parent, device):

        tk.Frame.__init__(self, parent)

        self.parent = parent
        self.font = parent.font

        # Device reference
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
            self.device.is_connected = True
            label.config(fg='green')
            txt.set("Disconnect")
        else:
            try:
                self.device.disconnect()
            except zmq.ZMQError:
                print ("Failed to disconnected from " + self.device.name + " endpoint.")

            self.device.is_connected = False
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

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)

        self.parent = parent
        self.font = parent.font

        # The connections we are interested in
        self.connections = [RemoteConnection(self, dev) for dev in DEVICES]

        self.initUI()

    def initUI(self):

        self.config(borderwidth=2, relief=tk.RAISED)
        self.pack()

        # Current row counter
        i = 0

        # Connection UIs
        l_connections = tk.Label(self, text="Remote Endpoints", font=self.font)
        l_connections.grid(row=i, column=0)
        i+=1

        start_row = i
        for j,c in enumerate(self.connections):
            i = start_row + j
            c.grid(row=i, column=0, pady=5, sticky="ew")
        i += 1
        
        b_frame = tk.Frame(self)

        # Record control buttons
        l_connections = tk.Label(self, text="Remote Controls", font=self.font)
        l_connections.grid(row=i, column=0)
        i+=1

        b_help = tk.Button(b_frame, text="Help", font=self.font, command=self.printHelp)
        b_start = tk.Button(b_frame, text="Start", font=self.font, command=self.startRecording)
        b_stop = tk.Button(b_frame, text="Stop", font=self.font, command=self.stopRecording)
        b_new = tk.Button(b_frame, text="New", font=self.font, command=self.makeNewFile)
        b_exit = tk.Button(b_frame, text="Exit", font=self.font, command=self.exitAll)

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

    def exitAll(self):
        for i, conn in enumerate(self.connections):
            if conn.device.is_connected:
                conn.device.exit()

def main():

    root = tk.Tk()
    root.title("Stupid Controller")
    root.font = tkFont.Font(family="Helvetica", size=12)
    root.geometry("250x150+300+300")
    app = RemoteControl(root)
    root.mainloop()

if __name__ == '__main__':
    main()
