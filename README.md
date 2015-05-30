## Real-time position tracker for animal behavior
Simple tracker consists of a set of programs for processing images, extracting position information, and streaming data to disk and the network that communicate through shared memory. This model enables quick, scripted construction of complex data processing chains without relying on a complicated GUI or plugin architecture.

###Installation

####Flycapture SDK (If point-grey camera is used)
- Go to [point-grey website](www.ptgrey.com)
- Download the FlyCapture2 SDK (version > 2.7.3)
- Extract the archive and use the `install_flycapture.sh` script to install the SDK on your computer.

```bash
tar xf flycapture.tar.gz
cd flycapture
sudo ./install_flycapture
```

#### Boost
```bash
wget http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.gz/download
tar -xf download
sudo cp -r boost_1_58_0 /opt
cd opt/boost_1_58_0/
sudo ./bootstrap.sh
sudo ./b2 --with-program_options --with_system --with_thread
```

####OpenCV
```bash
wget https://github.com/Itseez/opencv/archive/3.0.0-rc1.zip -O opencv.zip
unzip opencv.zip -d opencv
cd opencv/opencv-3.0.0-rc1 
mkdir release
cd release
cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local ..
make
sudo make install
```

####RapidJSON and cpptoml
Starting in the simple-tracker root directory,
```bash
cd lib
./updatelibs.sh
```

###tmux (for running scripts in playpen)
```bash
sudo apt-get install tmux 
```

### Manual

Simple tracker consists of a set of programs that communicate through shared memory to capture, process, and record video streams. Simple tracker works with two basic data types: `frames` and `positions`. 

* `frame`: a thread safe, shared-memory abstraction of a [cv::Mat object](http://docs.opencv.org/modules/core/doc/basic_structures.html#mat).
* `position`: a thread safe 2D position object.

Each component of the simple-tracker project is an executable defined by its input/output signature. Here is each component, with a corresponding IO signature. Below, the signature, usage information, example usage, and configuration options are provided for each component.

#### frameserve
Video frame server. Serves video streams to named shard memory from physical devices (e.g. webcam or gige camera) or from disk.

##### Signature
```
<div align="center">
┌────────────┐          
│ frameserve │ ──> frame
└────────────┘          
</div>
```

##### Usage
```
Usage: frameserve [OPTIONS]
   or: frameserve TYPE SINK [CONFIGURATION]

SINK
  The name of the memory segment to stream to.

TYPE
  'wcam': Onboard or USB webcam.
  'gige': Point Grey GigE camera.
  'file': Stream video from file.

OPTIONS:
  --help                    Produce help message.
  -v [ --version ]          Print version information.

CONFIGURATION:
  -c [ --config-file ] arg  Configuration file.
  -k [ --config-key ] arg   Configuration key.
  -f [ --video-file ] arg   Path to video file if 'file' is selected as the 
                            server TYPE.
```

##### Example
```bash
# For example
frameserve wcam wraw 
frameserve gige graw -c config.toml -k gige_config
frameserve file fraw -f ./video.mpg -c config.toml -k file_config
```

##### Configuration

`TYPE=gige`

- `index [+int]` User specified camera index. Useful in multi-camera imaging configurations.
- `exposure [float]` Automatically adjust both shutter and gain to achieve given exposure. Specified in dB.
- `shutter [+int]` Shutter time in milliseconds. Specifying `exposure` overrides this option.
- `gain [float]` Sensor gain value. Specifying `exposure` overrides this option.
- `white_bal [{+int, +int}]`
- `roi [{+int, +int, +int, +int}]`
- `trigger_on [bool]`
- `triger_polarity [bool]`
- `trigger_mode [+int]`

`TYPE=file`

- `frame_rate [float]` Frame rate in frames per second

#### viewer
Video frame viewer. Displays video stream from named shard memory.

##### Signature
```
<div align="center">
          ┌────────┐          
frame ──> │ viewer │          
		  └────────┘          
</div>
```

### TODO
- [x] Interprocess data processing synchronization
    - Whatever is chosen, all subsequent processing must propagate in accordance with the frame captured by the base image server(s).
    - e.g.
```
        Camera --> Background Subtract --> Detector --> Decorator --> Viewer
               ╲                                      ╱          ╲
	             ------------------------------------              -> Recorder    	
```
    - In this case, Decorator must block until Detector provides a result. However, Camera _may_ have produced another image in the meantime causing the Detector result and the image used by the Decorator to be out of sync. I need to find an intelligent way to deal with this.
    - **Edit**: Ended up using several autonomous semaphores which (1) keep track of the number of clients attached to each server and (2) enforce synchronized publication and read events on the server and client(s) respectively. These semaphores implement two canonical synchronization patters: the `rendezvous point` and the `turnstile`.
- [x] Start synchronization. The data processing chain should be initialized and waiting before the first image get sent.
    - This is optional. The current scheme allows servers to be started before clients and clients to be added while the servers are running
    - If no clients are attached to the server, the server will not bother to buffer data
    - Clients can also be removed while the server is running.
    - If clients are started before servers, then starts will be synchronized and no samples will be lost
- [x] Frame buffer?
    - This is now an intrinsic property of all data server classes and class templates (SMServer and MatServer)
- [x] IPC method?
    - ~~UPD, TCP,~~ **shared memory**, ~~pipe?~~
- [ ] Networked communication with clients that use extracted positional information
    - Wire format: per packet, one time-stamp and N frames labeled by camera serial number. Frames encoded to something like rgb8 char array
        - Strongly prefer to consume JSON over something ad hoc, opaque and untyped
    - Multiple clients
        - Broadcast over UDP
        - Shared memory (no good for remote tracker)
        - TCP/IP with thread for each client 
- [x] General C++ coding practice
    - Pass by const ref whenever possible. Especially relevant when passing derived objects to prevent slicing.
    - const member properties can be initialized in the initialization list, rather than assigned in the constructor body. Take advantage.
- [x] Implement pure intensity based detector (now color conversion, just saturation on raw image)
    - EDIT: This is just a special case of the already implemented H<b>S</b>V detector.
- [x] Implement position Filter (Kalman is first implementation)
- [ ] Implement recorder (Position and images? Viewer can also record?)
   - Viewer can record snapshots using keystroke. True recording would be a side effect.
   - Recorder will record up to N positions and N video streams. Of course, recorders can be paralleled to decrease computation burden on any single instance.
- [x] Camera configuration should specify frame capture due to digital pulses on a user selected GPIO line or free running.
- [x] To simplify IPC, clients should copy data in guarded sections. This limits the amount of time locks are engaged and likely, esp for cv::mat's make up for the copy in the increased amount of code that can be executed in parallel.
- [x] Can image metadata be packaged with shared cv::mats?
    - Frame rate
    - pixel -> cm transformation information
    - Sample number
- [x] Camera class should implement distortion correction (see [this example](https://github.com/Itseez/opencv/blob/6df1198e8b1ea4925cbce943a1dc6549f27d8be2/modules/calib3d/test/test_fisheye.cpp))
    - Ended up just hacking together a dedicated executable to produce the camera matrix and distortion parameters. Its called calibrate and its in the camserv project.
- [ ] Cmake improvements
    - Global build script to make all of the programs in the project
	- CMake managed versioning
- [ ] Travis CI
    - Get it building using the improvements to CMake stated in last TODO item
- [ ] Dealing with dropped frames
    - Right now, I poll the camera for frames. This is fine for a file, but not necessarily for a physical camera whose acquisitions is governed by an external, asynchronous clock
    - Instead of polling, I need an event driven frame server. In the case of a dropped frame, the server __must__ increment the sample number to prevent offsets from occurring.

#### Connecting to point-grey PGE camera in Linux
- First you must assign your camera a static IP address. 
  - The easiest way to do this is to use a Windows machine to run the the IP configurator program provided by Point Grey.
- The ipv4 method should be set to __manual__.
- Finally, you must the PG POE gigabit interface to (1) have the same network prefix and (2) be on the same subnet as your Gigabit camera. 
  - For instance, assume that your camera was assigned the following private ipv4 configuration:
    - Camera IP: 192.168.0.1
    - Subnet mask: 255.255.255.0
  - In this case, a functional ipv4 configuration for the POE Gigabit Ethernet card in the host PC could be:
    - POE gigabit card IP: 192.168.0.100
    - Subnet mask: 255.255.255.0
    - DNS server IP: 192.168.1.1
- Note that if you want to add another network interface for another camera, it must exist on a separate subnet!    - For instance, we could repeat the above configuration steps for the second camera using the following settings:
    - Camera IP: 192.168.1.1
    - Subnet mask: 255.255.255.0
  - In this case, a functional ipv4 configuration for the POE Gigabit Ethernet card in the host PC could be:
    - POE gigabit card IP: 192.168.__1__.100
    - Subnet mask: 255.255.255.0
    - DNS server IP: 192.168.1.1
- Finally, you must enable jumbo frames on the network interface
  - Assume that the camera is using eth2
  - `sudo ifconfig eth2 mtu 9000` 


