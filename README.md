__Oat__ is a set of programs for processing images, extracting object
position information, and streaming data to disk and the network in real-time.
Oat subcommands are independent programs that each perform a single operation. 
However, programs can communicate through shared memory. This allows a user 
to chain operations together in arrangements suitable for particular context or 
tracking requirement. This architecture enables quick, scripted construction of 
custom data processing chains. Oat is primarily used for real-time animal 
position tracking in the context of experimental neuroscience, but can be used 
in any circumstance that requires real-time object tracking.

### Manual
Oat components are a set of programs that communicate through shared memory to 
capture, process, and record video streams. Oat components act on two basic 
data types: `frames` and `positions`. 

* `frame` - a shared-memory abstraction of a 
  [cv::Mat object](http://docs.opencv.org/modules/core/doc/basic_structures.html#mat).
* `position` - 2D position type.

Oat components can be chained together to execute data processing pipelines, 
with individual components executing largely in parallel. Processing pipelines 
can be split and merged while maintaining thread-safety and sample synchronization. 
For example, a script to detect the position of a single object in pre-recorded 
video file might look like this:

```bash
# Serve frames from a video file to the 'raw' stream
oat frameserve file raw -f ./video.mpg &

# Perform background subtraction on the 'raw' stream 
# Serve the result to the 'filt' stream
oat framefilt bsub raw filt &

# Perform color-based object detection on the 'filt' stream
# Serve the object positionto the 'pos' stream
oat detect hsv filt pos &

# Decorate the 'raw' stream with the detected position form the `pos` stream
# Serve the decorated images to the 'dec' stream
oat decorate -p pos raw dec &

# View the 'dec' stream
oat view dec &

# Record the 'dec' and 'pos' streams to file in the current directory
oat record -i dec -p pos -f ./
```

This script has the following graphical representation:
```
frameserve ──> framefilt ──> posidet ──> decorate ───> view
           ╲                           ╱         ╲
             ─────────────────────────             ──> record   	
```

Generally, an Oat component is called in the following pattern:
```
oat <subcommand> [TYPE] [IO] [CONFIGURATION]
```
`subcommand` indicates the component that will be executed. Components
are classified according to their type signature. For instance, `framefilt` 
(frame filter) accepts a frame and produces a frame. `posifilt` (position filter) 
accepts a position and produces a position. `frameserve` (frame server) produces 
a frame, and so on.  The `TYPE` parameter specifies a concrete type of transform 
(e.g. for the framefilt subcommand this could be `bsub` for background subtraction). 
The `IO` specification indicates where the component is receiving data from 
and to where the processed data should be published. The `CONFIGURATION` 
specification is used to provide parameters to the component. Below, the type 
signature, usage information, examples, and configuration options are provided 
for each Oat component.


#### `frameserve`
Video frame server. Serves video streams to named shared memory from physical 
devices (e.g. webcam or GIGE camera) or from disk.

##### Signature
```
┌────────────┐          
│ frameserve │ ──> frame
└────────────┘          
```

##### Usage
```
Usage: frameserve [INFO]
   or: frameserve TYPE SINK [CONFIGURATION]

SINK
  The name of the memory segment to stream to.

TYPE
  wcam: Onboard or USB webcam.
  gige: Point Grey GigE camera.
  file: Stream video from file.

INFO:
  --help                    Produce help message.
  -v [ --version ]          Print version information.

CONFIGURATION:
  -c [ --config-file ] arg  Configuration file.
  -k [ --config-key ] arg   Configuration key.
  -f [ --video-file ] arg   Path to video file if 'file' is selected as the 
                            server TYPE.
```

##### Examples
```bash
# Serve to the 'wraw' stream from a webcam 
oat frameserve wcam wraw 

# Stream to the 'graw' stream from a point-grey GIGE camera
# using the gige_config tag from the config.toml file
oat frameserve gige graw -c config.toml -k gige_config

# Serve to the 'fraw' stream from a previously recorded file
# using the file_config tag from the config.toml file
oat frameserve file fraw -f ./video.mpg -c config.toml -k file_config
```

##### Configuration

###### `TYPE=gige`

- `index [+int]` User specified camera index. Useful in multi-camera imaging configurations.
- `exposure [float]` Automatically adjust both shutter and gain to achieve given exposure. Specified in dB.
- `shutter [+int]` Shutter time in milliseconds. Specifying `exposure` overrides this option.
- `gain [float]` Sensor gain value. Specifying `exposure` overrides this option.
- `white_bal [{+int, +int}]`
- `roi [{+int, +int, +int, +int}]`
- `trigger_on [bool]`
- `triger_polarity [bool]`
- `trigger_mode [+int]`

###### `TYPE=file`

- `frame_rate [float]` Frame rate in frames per second
- `roi [{+int, +int, +int, +int}]`



#### `framefilt`
Frame filter.

##### Signature
```
         ┌───────────┐          
frame──> │ framefilt │ ──> frame
         └───────────┘          
```

#### `view`
Frame viewer. Displays video stream from named shard memory on a mointor. Also 
permits the user to take snapshots of the viewed stream by pressing <kbd>s</kbd>
while the display window is in focus.

##### Signature
```
          ┌──────┐
frame ──> │ view │
          └──────┘
```

##### Usage
```
Usage: view [INFO]
   or: view SOURCE [CONFIGURATION] 

INFO:
  --help                 Produce help message.
  -v [ --version ]       Print version information.

CONFIGURATION:
  -n [ --filename ] arg  The base snapshot file name.
                         The timestamp of the snapshot will be prepended to 
                         this name.
                		 If not provided, the SOURCE name will be used.
  -f [ --folder ] arg    The folder to which snapshots will be saved

NOTE:
  To take a snapshot of the currently displayed frame, press 's' key while 
  the display window is in focus.
```

##### Example
```bash
# View frame stream named raw
oat view raw 

# View frame stream named raw and specify that snapshots should be saved
# to the Desktop with base name 'snapshot'
oat view raw -f ~/Desktop -n snapshot
```

#### `posidet`
Position detector. Detects object position within a frame stream using one of 
several methods.

##### Signature
```
          ┌─────────┐
frame ──> │ posidet │ ──> position
          └─────────┘
```

##### Example
```bash
# Use color-based object detection on the 'raw' frame stream 
# publish the result to the 'cpos' position stream
# Use detector settings supplied by the hsv_config tag in config.toml
oat posidet hsv raw cpos -c config.toml -k hsv_config

# Use motion-based object detection on the 'raw' frame stream 
# publish the result to the 'mpos' position stream
oat posidet diff raw mpos  
```

#### `posifilt`
Position filter. Filters position stream to, for example, remove discontinuities 
due to noise or discontinuities in position detection. 

##### Signature
```
             ┌──────────┐
position ──> │ posifilt │ ──> position
             └──────────┘
```

##### Usage
TODO

#### `posicom`
Position combiner. Combines position inputs according to a specified operation.

##### Signature
```
               ┌────────┐
position 0 ──> │        │
position 1 ──> │        │
  :            │posicom │ ──> position
position N ──> │        │
               └────────┘
```

#### `decorate`
Frame decorator. Annotates frames with sample times, dates, and/or positional 
information.

##### Signature
```
               ┌─────────┐
     frame ──> │         │
position 0 ──> │         │
position 1 ──> │decorate │ ──> frame
  :            │         │
position N ──> │         │
               └─────────┘
```

##### Usage
TODO

#### `record`
Stream recorder. Saves frame and positions streams to disk. 

* `frame` streams are compressed and saved as individual video files (
  [H.264](http://en.wikipedia.org/wiki/H.264/MPEG-4_AVC) compression format AVI file).
* `position` streams are combined into a single [JSON](http://json.org/) file. 
  Position files have the following structure:

```javascript
{header:  [ {timestamp: YYYY-MM-DD-hh-mm-ss}, 
            {sample_rate_hz: Fs}, 
            {sources: [s1, s2, ..., sN]}                   ] }
{samples: [ [0, {Position1}, {Position2}, ..., {PositionN} ],
            [1, {Position1}, {Position2}, ..., {PositionN} ],
             :
            [T, {Position1}, {Position2}, ..., {PositionN} ]] }
```
where each position object is defined as:

```javascript
{TODO}
```

All streams are saved with a single recorder have the same base file name and save location (see usage). Of course, multiple recorders can be used in parallel to (1) parallelize the computational load of video compression, which tends to be quite intense and (2) save to multiple locations simultaneously.

##### Signature
```
               ┌────────┐
position 0 ──> │        │
position 1 ──> │        │
  :            │        │
position N ──> │        │
               │ record │
   frame 0 ──> │        │
   frame 1 ──> │        │
     :         │        │
   frame N ──> │        │
               └────────┘
```

##### Usage
```
Usage: record [OPTIONS]
   or: record [CONFIGURATION]

OPTIONS:
  --help                        Produce help message.
  -v [ --version ]              Print version information.

CONFIGURATION:
  -n [ --filename ] arg         The base file name to which to source name will
                                be appended
  -f [ --folder ] arg           The path to the folder to which the video 
                                stream and position information will be saved.
  -d [ --date ]                 If specified, YYYY-MM-DD-hh-mm-ss_ will be 
                                prepended to the filename.
  -p [ --positionsources ] arg  The name of the server(s) that supply object 
                                position information.The server(s) must be of 
                                type SMServer<Position>
                                
  -i [ --imagesources ] arg     The name of the server(s) that supplies images 
                                to save to video.The server must be of type 
                                SMServer<SharedCVMatHeader>

```

##### Example
```bash
# Save positional stream 'pos' to current directory
oat record -p pos 

# Save positional stream 'pos1' and 'pos2' to current directory
oat record -p pos1 pos2

# Save positional stream 'pos1' and 'pos2' to Desktop directory and 
# prepend the timestamp to the file name
oat record -p pos1 pos2 -d -f ~/Desktop

# Save frame stream 'raw' to current directory
oat record -i raw

# Save frame stream 'raw' and positional stream 'pos' to Desktop 
# directory and prepend the timestamp and 'my_data' to each filename
oat record -i raw -p pos -d -f ~/Desktop -n my_data

```

### Installation

#### Flycapture SDK (If point-grey camera is used)
- Go to [point-grey website](www.ptgrey.com)
- Download the FlyCapture2 SDK (version > 2.7.3)
- Extract the archive and use the `install_flycapture.sh` script to install the SDK on your computer.

```bash
tar xf flycapture.tar.gz
cd flycapture
sudo ./install_flycapture
```

#### [Boost](http://www.boost.org/)
```bash
wget http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.gz/download
tar -xf download
sudo cp -r boost_1_58_0 /opt
cd opt/boost_1_58_0/
sudo ./bootstrap.sh
sudo ./b2 --with-program_options --with_system --with_thread
```

#### [OpenCV](http://opencv.org/)

__Note__: OpenCV must be installed with ffmpeg support in order for offline analysis 
of pre-recorded videos to occur at arbitrary frame rates. If it is not, gstreamer 
will be used to serve from video files at the rate the files were recorded. No cmake 
flags are required to configure the build to use ffmpeg. OpenCV will be built 
with ffmpeg support if something like
```bash
-- FFMPEG:          YES
-- codec:           YES (ver 54.35.0)
-- format:          YES (ver 54.20.4)
-- util:            YES (ver 52.3.0)
-- swscale:         YES (ver 2.1.1)
```
appears in the cmake output text. 

__Note__: To increase Oat's video visualization performance using `oat view`, you can 
build OpenCV with OpenGL support. This will open up significant processing bandwidth 
to other Oat components and make for faster processing pipelines. To compile OpenCV
with OpenGL support, add the `-DWITH_OPENGL=ON` flag in the cmake command below.
OpenCV will be build with OpenGL support if `OpenGL support: YES` appears in the 
cmake output text.

__Note__: If you have [NVIDA GPU that supports CUDA](https://developer.nvidia.com/cuda-gpus), 
you can build OpenCV with CUDA support to enable GPU accelerated video processing. 
To do this, will first need to install the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit). 
Be sure to read the [installation instructions](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/index.html) 
since it is a multistep process.To compile OpenCV with CUDA support, add the 
`-DWITH_CUDA=ON` flag in the cmake command below.

```bash
# Install dependencies
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
sudo ldconfig -v

# Get OpenCV
wget https://github.com/Itseez/opencv/archive/3.0.0-rc1.zip -O opencv.zip
unzip opencv.zip -d opencv

# Build OpenCV
cd opencv/opencv-3.0.0-rc1 
mkdir release
cd release
# Add -DWITH_CUDA=ON for CUDA support and -DWITH_OPENGL for OpenGL support 
cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local ..
make
sudo make install
```

#### [RapidJSON](https://github.com/miloyip/rapidjson) and [cpptoml](https://github.com/skystrife/cpptoml)
Starting in the project root directory,
```bash
cd lib
./updatelibs.sh
```

### TODO
- [ ] Networked communication with clients that use extracted positional information
    - Strongly prefer to consume JSON over something ad hoc, opaque and untyped
    - Multiple clients
        - Broadcast over UDP
        - Shared memory (no good for remote tracker)
        - TCP/IP with thread for each client 
- [ ] Cmake improvements
    - ~~Global build script to make all of the programs in the project~~
    - ~~CMake managed versioning~~
    - Output messages detailing required and recommended pacakges.
    - Windows build?
- [ ] Travis CI
    - Get it building using the improvements to CMake stated in last TODO item
- [ ] Dealing with dropped frames
    - Right now, I poll the camera for frames. This is fine for a file, but not necessarily for a physical camera whose acquisitions is governed by an external, asynchronous clock
    - Instead of polling, I need an event driven frame server. In the case of a dropped frame, the server __must__ increment the sample number, even if it does not serve the frame, to prevent offsets from occurring.
- [ ] shmem type checking by clients, exit gracefully in the case of incorrect type
   - e.g. a framefilter tries to use a position filter as a SOURCE. In this case, the framefilter needs to realize that the SOURCE type is wrong and exit.
- [ ] Exception saftey for all components
    - frameserve
    - ~~framefilt~~
    - ~~posidet~~
    - ~~posicom~~
    - ~~posifilt~~
    - ~~positest~~
    - ~~record~~
    - ~~view~~
    - ~~decorate~~
- [ ] Use PMPL for library classes to hide implementation (private member properties/functions)

####  Setting up a Point-grey PGE camera in Linux
`oat frameserve` supports using Point Grey GIGE cameras to collect frames. I found
the setup process to be straightforward and robust, but only after cobling together
the following notes:

- First you must assign your camera a static IP address. 
    - The easiest way to do this is to use a Windows machine to run the the IP 
      configurator program provided by Point Grey. If someone has a way to do this
      without Windows, please tell me.
- The ipv4 method should be set to __manual__.
- Finally, you must the PG POE gigabit interface to (1) have the same network prefix and (2) be on the same subnet as your Gigabit camera. 
    - For instance, assume that your camera was assigned the following private ipv4 configuration:
        - Camera IP: 192.168.0.1
        - Subnet mask: 255.255.255.0
    - In this case, a functional ipv4 configuration for the POE Gigabit Ethernet card in the host PC could be:
        - POE gigabit card IP: 192.168.0.100
        - Subnet mask: 255.255.255.0
        - DNS server IP: 192.168.1.1
- Note that if you want to add another network interface for another camera, it must exist on a separate subnet!    
    - For instance, we could repeat the above configuration steps for the second camera using the following settings:
        - Camera IP: 192.168.1.1
        - Subnet mask: 255.255.255.0
    - In this case, a functional ipv4 configuration for the POE Gigabit Ethernet card in the host PC could be:
        - POE gigabit card IP: 192.168.__1__.100
        - Subnet mask: 255.255.255.0
        - DNS server IP: 192.168.1.1
- Next, you must enable jumbo frames on the network interface
   - Assume that the camera is using eth2
   - `sudo ifconfig eth2 mtu 9000` 
- Finally, increase the amount of memory Linux uses for receive buffers using the sysctl interface
    - `sudo sysctl -w net.core.rmem_max=1048576 net.core.rmem_default=1048576`
	- _Note_: In order for these changes to persist after system reboots, the following lines must be manually added to the bottom of the /etc/sysctl.conf file:
	- net.core.rmem_max=1048576
	- net.core.rmem_default=1048576



