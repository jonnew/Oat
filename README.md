__Oat__ is a set of programs for processing images, extracting object position
information, and streaming data to disk and/or the network in real-time.  Oat
subcommands are independent programs that each perform a single operation but
that can communicate through shared memory. This allows a user to chain
operations together in arrangements suitable for particular context or tracking
requirement. This architecture enables scripted construction of custom data
processing chains. Oat is primarily used for real-time animal position tracking
in the context of experimental neuroscience, but can be used in any
circumstance that requires real-time object tracking.

### Manual
Oat components are a set of programs that communicate through shared memory to
capture, process, perform object detection within, and record video streams.
Oat components act on two basic data types: `frames` and `positions`. 

* `frame` - Video frame.
* `position` - 2D position.

Oat components are be chained together to realize custom data processing
pipelines, with individual components executing largely in parallel. Processing
pipelines can be split and merged while maintaining thread-safety and sample
synchronization.  For example, a script to detect the position of a single
object in pre-recorded video file might look like this:

```bash
# Serve frames from a video file to the 'raw' stream
oat frameserve file raw -f ./video.mpg &

# Perform background subtraction on the 'raw' stream 
# Serve the result to the 'filt' stream
oat framefilt bsub raw filt &

# Perform color-based object position detection on the 'filt' stream 
# Serve the object positionto the 'pos' stream
oat posidet hsv filt pos &

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
The `<subcommand>` indicates the component that will be executed. Components
are classified according to their type signature. For instance, `framefilt`
(frame filter) accepts a frame and produces a frame. `posifilt` (position
filter) accepts a position and produces a position. `frameserve` (frame server)
produces a frame, and so on.  The `TYPE` parameter specifies a concrete type of
transform (e.g. for the framefilt subcommand this could be `bsub` for
background subtraction).  The `IO` specification indicates where the component
will receive data from and to where the processed data should be published. The
`CONFIGURATION` specification is used to provide parameters to shape the
component's operation.  Aside from command line options and switches, which are
listed using the `--help` option for each subcommand, the user can often
provide an external file containing a configuration table to pass parameters to
a component. Some configuration parameters can only be specified using a
configuration file.  Configuration files are written in plain text using
[TOML](https://github.com/toml-lang/toml). A multi-component processing script
can share a configuration file because each component accesses parameter
information using a file/key pair, like so

```toml
[key]
parameter_0 = 1                 # Integer
parameter_1 = true              # Boolean
parameter_2 = 3.14              # Double
parameter_3 = [1.0, 2.0, 3.0]   # Array of doubles
```

or more concretely,

```toml
# Example configuration file for frameserve ──> framefilt
[frameserve-par]
frame_rate = 30                 # FPS
roi = [10, 10, 100, 100]        # Region of interest

[framefilt-par]
mask = "~/Desktop/mask.png"     # Path to mask file
```

The type and sanity of parameter values are checked by Oat before they are
used. Below, the type signature, usage information, available configuration
parameters, examples, and configuration options are provided for each Oat
component.


#### Frame server
`oat-frameserve` - Serves video streams to shared memory from physical devices
(e.g. webcam or GIGE camera) or from file.

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
  User-supplied name of the memory segment to publish frames to (e.g. raw).

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

##### Configuration file options

TYPE=`gige`

- `index [+int]` User specified camera index. Useful in multi-camera imaging
  configurations.
- `exposure [float]` Automatically adjust both shutter and gain to achieve
  given exposure. Specified in dB.
- `shutter [+int]` Shutter time in milliseconds. Specifying `exposure`
  overrides this option.
- `gain [float]` Sensor gain value. Specifying `exposure` overrides this
  option.
- `white_bal [{+int, +int}]` White balance in [red blue] intensity value.
- `roi [{+int, +int, +int, +int}]` Region of interest to extract from the
  camera or video stream specified as `[x_offset, y_offset, width, height]`
- `trigger_on [bool]` True to use camera trigger, false to use software
  polling.
- `triger_polarity [bool]` True to trigger on rising edge, false to trigger on
  falling edge.
- `trigger_mode [+int]` Point-grey trigger mode. Common values are 7 and 14 for
  [TODO]

TYPE=`file`

- `frame_rate [float]` Frame rate in frames per second
- `roi [{+int, +int, +int, +int}]` Region of interest to extract from the
  camera or video stream specified as `[x_offset, y_offset, width, height]`

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

#### Frame filter
`oat-framefilt` - Receive frames from named shared memory, filter, and publish
to a second memory segment. Generally, used to pre-process frames prior to
object position detection. For instance, `framefilt` could be used to perform
background subtraction or application of a mask to isolate a region of
interest.

##### Signature
```
          ┌───────────┐          
frame ──> │ framefilt │ ──> frame
          └───────────┘          
```

##### Usage
```
Usage: framefilt [INFO]
   or: framefilt TYPE SOURCE SINK [CONFIGURATION]
Filter frames from SOURCE and published filtered frames to SINK.
TYPE
  bsub: Background subtraction
  mask: Binary mask

SOURCE:
  User-supplied name of the memory segment to receive frames from (e.g. raw).

SINK:
  User-supplied name of the memory segment to publish frames to (e.g. filt).

INFO:
  --help                    Produce help message.
  -v [ --version ]          Print version information.

CONFIGURATION:
  -c [ --config-file ] arg  Configuration file.
  -k [ --config-key ] arg   Configuration key.
  -m [ --invert-mask ]      If using TYPE=mask, invert the mask before applying
```

##### Configuration file options

TYPE=`bsub`

- `background [string]` Path to a background image to be subtracted from the
  SOURCE frames. This image must have the same dimensions as frames from
  SOURCE.

TYPE=`mask`

- `mask [string]` Path to a binary image used to mask frames from SOURCE.
  SOURCE frame pixels with indices corresponding to non-zero value pixels in
  the mask image will be unaffected. Others will be set to zero. This image
  must have the same dimensions as frames from SOURCE.

##### Examples
```bash
# Receive frames from 'raw' stream
# Perform background subtraction using the first frame as the background
# Publish result to 'sub' stream
oat framefilt bsub raw sub

# Receive frames from 'raw' stream
# Apply a mask specified in a configuration file
# Publish result to 'roi' stream
oat framefilt mask raw roi -c config.toml -k mask-config
```

#### Viewer
`oat-view` - Receive frames from named shared memory and display them on a 
monitor. Additionally, allow the user to take snapshots of the currently
displayed frame by pressing <kbd>s</kbd> while the display window is
in focus.

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
Display frame SOURCE on a monitor.

SOURCE:
  User-supplied name of the memory segment to receive frames from (e.g. raw).

INFO:
  --help                 Produce help message.
  -v [ --version ]       Print version information.

CONFIGURATION:
  -n [ --filename ] arg  The base snapshot file name.
                         The timestamp of the snapshot will be prepended to 
                         this name.If not provided, the SOURCE name will be 
                         used.
                         
  -f [ --folder ] arg    The folder in which snapshots will be saved.
```

##### Example
```bash
# View frame stream named raw
oat view raw 

# View frame stream named raw and specify that snapshots should be saved
# to the Desktop with base name 'snapshot'
oat view raw -f ~/Desktop -n snapshot
```

#### Position detector
`oat-posidet` - Receive frames from named shared memory and perform object
position detection within a frame stream using one of several methods. Publish
detected positions to a second segment of shared memory.

##### Signature
```
          ┌─────────┐
frame ──> │ posidet │ ──> position
          └─────────┘
```

##### Usage
```
Usage: posidet [INFO]
   or: posidet TYPE SOURCE SINK [CONFIGURATION]
Perform object position detection on frames from SOURCE.
Publish detected object positions to SINK.

TYPE
  diff: Difference detector (grey-scale, motion)
  hsv : HSV detector (color)

SOURCE:
  User-supplied name of the memory segment to receive 
  frames from (e.g. raw).

SINK:
  User-supplied name of the memory segment to publish 
  detected positions to (e.g. pos).

INFO:
  --help                    Produce help message.
  -v [ --version ]          Print version information.

CONFIGURATION:
  -c [ --config-file ] arg  Configuration file.
  -k [ --config-key ] arg   Configuration key.
```

##### Configuration file options

TYPE=`hsv`
- `tune` [bool] Provide sliders for tuning hsv parameters
- `erode` [+int] Candidate object erosion kernel size (pixels)
- `dilate` [+int] Candidate object dilation kernel size (pixels)
- `min_area` [+int] Minimum object area (pixels^2)
- `max_area` [+int] Maximum object area (pixels^2)
- `h_thresholds` = {min [+int], max [+int]} Hue pass band
- `s_thresholds` = {min [+int], max [+int]} Saturation pass band 
- `v_thresholds` = {min [+int], max [+int]} Value pass band

TYPE=`diff`
- `tune [bool] Provide sliders for tuning diff parameters
- `blur` [+int] Blurring kernel size (normalized box filter; pixels)
- `diff_threshold` [+int] Intensity difference threshold 

##### Example
```bash
# Use color-based object detection on the 'raw' frame stream 
# publish the result to the 'cpos' position stream
# Use detector settings supplied by the hsv_config key in config.toml
oat posidet hsv raw cpos -c config.toml -k hsv_config

# Use motion-based object detection on the 'raw' frame stream 
# publish the result to the 'mpos' position stream
oat posidet diff raw mpos  
```

#### Position filter
`oat-posifilt` - Receive positions from named shared memory, filter, and
publish to a second memory segment. Can be used to, for example, remove
discontinuities due to noise or discontinuities in position detection with a
Kalman filter or annote categorical postion information based on user supplied
region contours.

##### Signature
```
             ┌──────────┐
position ──> │ posifilt │ ──> position
             └──────────┘
```

##### Usage
```
Usage: posifilt [INFO]
   or: posifilt TYPE SOURCE SINK [CONFIGURATION]
Filter positions from SOURCE and published filtered positions to SINK.

TYPE
  kalman: Kalman filter
  homo: homography transform
  region: position region annotation

SOURCE:
  User-supplied name of the memory segment to receive positions from (e.g. rpos).

SINK:
  User-supplied name of the memory segment to publish positions to (e.g. rpos).

INFO:
  --help                    Produce help message.
  -v [ --version ]          Print version information.

CONFIGURATION:
  -c [ --config-file ] arg  Configuration file.
  -k [ --config-key ] arg   Configuration key.
```

##### Configuration file options

TYPE=`kalman`
- `dt` [+float] Sample period (seconds)
- `timeout` [+float] Seconds to perform position estimation detection with lack
  of updated position measure
- `sigma_accel` [+float] Standard deviation of normally distributed, random
  accelerations used by the internal model of object motion (Position
  units/s^2; e.g. Pixels/s^2)
- `sigma_noise` [+float] Standard deviation of randomly distributed position
  measurement noise (Position units; e.g. Pixels)
- `tune` [bool] Use the GUI to tweak parameters

TYPE=`homo`
- `homography` [[+float], [+float], [+float], Homography matrix for 2D position
                [+float], [+float], [+float],
                [+float], [+float], [+float]]
TYPE=`region` //TODO
CN = [[336.00, 272.50],
      [290.00, 310.00],
      [289.00, 369.50],
      [332.67, 417.33],
      [389.33, 413.33],
      [430.00, 375.33],
      [433.33, 319.33],
      [395.00, 272.00]]

R0 = [[654.00, 380.00],
      [717.33, 386.67],
      [714.00, 316.67],
      [655.33, 319.33]]

##### Example
```bash
# Use color-based object detection on the 'raw' frame stream 
# publish the result to the 'cpos' position stream
# Use detector settings supplied by the hsv_config key in config.toml
oat posidet hsv raw cpos -c config.toml -k hsv_config

# Use motion-based object detection on the 'raw' frame stream 
# publish the result to the 'mpos' position stream
oat posidet diff raw mpos  
```

TYPE=`hsv`
- `tune` [bool] Provide sliders for tuning hsv parameters
- `erode` [+int] Candidate object erosion kernel size (pixels)
- `dilate` [+int] Candidate object dilation kernel size (pixels)
- `min_area` [+int] Minimum object area (pixels^2)
- `max_area` [+int] Maximum object area (pixels^2)
- `h_thresholds` = {min [+int], max [+int]} Hue pass band
- `s_thresholds` = {min [+int], max [+int]} Saturation pass band 
- `v_thresholds` = {min [+int], max [+int]} Value pass band

TYPE=`diff`
- `tune [bool] Provide sliders for tuning diff parameters
- `blur` [+int] Blurring kernel size (normalized box filter; pixels)
- `diff_threshold` [+int] Intensity difference threshold 

##### Example
```bash
# Use color-based object detection on the 'raw' frame stream 
# publish the result to the 'cpos' position stream
# Use detector settings supplied by the hsv_config key in config.toml
oat posidet hsv raw cpos -c config.toml -k hsv_config

# Use motion-based object detection on the 'raw' frame stream 
# publish the result to the 'mpos' position stream
oat posidet diff raw mpos  
```

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
    - Option for building with/without point-grey support
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
- [ ] Use smart pointers to ensure proper resource management.
    - ~~Start with `recorder`, which has huge amounts of heap allocated objects pointed to with raw
      pointers~~
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



