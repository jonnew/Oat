__Oat__ is a set of programs for processing images, extracting object position
information, and streaming data to disk and/or the network in real-time.  Oat
subcommands are independent programs that each perform a single operation but
that can communicate through shared memory. This allows a user to chain
operations together in arrangements suitable for particular context or tracking
requirement. This architecture enables scripted construction of custom data
processing chains. Oat is primarily used for real-time animal position tracking
in the context of experimental neuroscience, but can be used in any
circumstance that requires real-time object tracking.

[![Build Status](https://travis-ci.org/jonnew/Oat.png?branch=master)](https://travis-ci.org/jonnew/Oat)

__Contributors__

- jonnew [http://www.mit.edu/~jpnewman/](http://www.mit.edu/~jpnewman/)

**Table of Contents**

- [Manual](#manual)
    - [Introduction](#introduction)
    - [Frame Server](#frame-server)
        - [Signature](#signature)
        - [Usage](#usage)
        - [Configuration Options](#configuration-options)
        - [Examples](#examples)
    - [Frame Filter](#frame-filter)
        - [Signature](#signature-1)
        - [Usage](#usage-1)
        - [Configuration Options](#configuration-options-1)
        - [Examples](#examples-1)
    - [Frame Viewer](#frame-viewer)
        - [Signature](#signature-2)
        - [Usage](#usage-2)
        - [Configuration Options](#configuration-options-2)
        - [Example](#example)
    - [Position Detector](#position-detector)
        - [Signature](#signature-3)
        - [Usage](#usage-3)
        - [Configuration Options](#configuration-options-3)
        - [Example](#example-1)
    - [Position Generator](#position-generator)
        - [Signature](#signature-4)
        - [Usage](#usage-4)
        - [Configuration Options](#configuration-options-4)
        - [Example](#example-2)
    - [Position Filter](#position-filter)
        - [Signature](#signature-5)
        - [Usage](#usage-5)
        - [Configuration Options](#configuration-options-5)
        - [Example](#example-3)
    - [Position Combiner](#position-combiner)
        - [Signature](#signature-6)
        - [Usage](#usage-6)
        - [Configuration Options](#configuration-options-6)
        - [Example](#example-4)
    - [Frame Decorator](#frame-decorator)
        - [Signature](#signature-7)
        - [Usage](#usage-7)
        - [Example](#example-5)
    - [Recorder](#recorder)
        - [Signature](#signature-8)
        - [Usage](#usage-8)
        - [Example](#example-6)
    - [Position Socket](#position-socket)
        - [Signature](#signature-9)
        - [Usage](#usage-9)
        - [Configuration Options](#configuration-options-7)
        - [Example](#example-7)
    - [Buffer](#buffer)
        - [Signatures](#signatures)
        - [Usage](#usage-10)
        - [Example](#example-8)
    - [Calibrate](#calibrate)
        - [Signature](#signature-10)
        - [Usage](#usage-11)
        - [Configuration Options](#configuration-options-8)
    - [Kill](#kill)
        - [Usage](#usage-12)
        - [Example](#example-9)
    - [Clean](#clean)
        - [Usage](#usage-13)
        - [Example](#example-10)
    - [Installation](#installation)
        - [Dependencies](#dependencies)
    - [Performance](#performance)
    - [Setting up a Point-grey PGE camera in Linux](#setting-up-a-point-grey-pge-camera-in-linux)
- [TODO](#todo)

\newpage
## Manual
### Introduction
Oat's design is influenced by the [UNIX
philosophy](https://en.wikipedia.org/wiki/Unix_philosophy), [suckless
tools](http://suckless.org/philosophy), and
[MEABench](http://www.danielwagenaar.net/res/software/meabench/). Oat consists
of a set of small, composable programs (called **components**). Components are
equipped with standard interfaces that permit communication through shared
memory to capture, process, and record video streams.  Currently, Oat
components act on two basic data types: `frames` and `positions`.

* `frame` - Video frame.
* `position` - 2D position.

Oat components can be chained together to realize custom dataflow networks that
operate on instances of the aforementioned datatypes, called **tokens**.  Token
processing pipelines can be split and merged while maintaining thread-safety
and sample synchronization.  The [messaging library](lib/shmemdf) underlying
the communication between Oat components has been optimized to reduce token
copying. For instance, `frame` passing is performed using a
[zero-copy](https://en.wikipedia.org/wiki/Zero-copy) protocol. This means that
passing `frames` between components in a user-configured processing network
incurs almost no memory or CPU cost compared to the
[monolithc](https://en.wikipedia.org/wiki/Monolithic_application) equivalent.
Further, great care was taken during implementations of Oat components to
minimize time spent in [critical
sections](https://en.wikipedia.org/wiki/Critical_section). This means that
individual components execute largely in parallel, even when components are
highly interdependent, facilitating efficient use of multi-core CPUs and
GPU-based processing acceleration.

To get a feel for how Oat is used, here is a script to detect the position of a
single object in pre-recorded video file:

```bash
# Serve frames from a video file to the 'raw' stream
oat frameserve file raw -f ./video.mpg &

# Perform background subtraction on the 'raw' stream
# Serve the result to the 'filt' stream
# If an appropriately configured GPU is available, this process will
# use it
oat framefilt mog raw filt &

# Perform color-based object position detection on the 'filt' stream
# Serve the object positionto the 'pos' stream. Allow parameter tuning
# through a simple GUI.
oat posidet hsv filt pos --tune &

# Decorate the 'raw' stream with the detected position form the `pos` stream
# Serve the decorated images to the 'dec' stream
oat decorate -p pos raw dec &

# View the 'dec' stream
oat view frame dec &

# Record the 'dec' and 'pos' streams to file in the current directory
oat record -i dec -p pos -f ./
```

This script has the following graphical representation:

```
frameserve ---> framefilt ---> posidet ---> decorate ---> view
           \                          \    /        \
             -----------------------------           ---> record
                                        \           /
                                          ---------
```

Generally, an Oat component is called in the following pattern:

```
oat <component> [TYPE] [IO] [CONFIGURATION]
```
The `<component>` indicates the component that will be executed. Components
are classified according to their type signature. For instance, `framefilt`
(frame filter) accepts a frame and produces a frame. `posifilt` (position
filter) accepts a position and produces a position. `frameserve` (frame server)
produces a frame, and so on.  The `TYPE` parameter specifies a concrete type of
transform (e.g. for the `framefilt` component, this could be `bsub` for
background subtraction). The `IO` specification indicates where the component
will receive data from and to where the processed data should be published. A
description of a component's purpose, its available TYPEs and correct IO
specification can be examined using the `--help` command line switch

```bash
oat <component> --help
```

The `CONFIGURATION` specification is used to provide parameters to shape the
component's operation and are TYPE-specific. Information on program options for
a particular concrete transform TYPE can be printed using

```bash
oat <component> <type> --help
```

In addition to command line input, all options can be specified using a
configuration file which is provided to the program using the `-c` command line
argument.

```
  -c [ --config ] arg             Configuration file/key pair.
                                  e.g. 'config.toml mykey'
```

For instance:

```bash
oat frameserve gige raw -c config.toml gige-config
```

The configuration file may contain many configuration tables that specify
options for multiple oat programs. These tables are addressed using a key
(`gige-config`) in the example above.  Configuration files are written in plain
text using [TOML](https://github.com/toml-lang/toml). A multi-component
processing script can share a configuration file because each component
accesses parameter information using a file/key pair, like so

```toml
[key]
parameter_0 = 1                 # Integer
parameter_1 = true              # Boolean
parameter_2 = 3.14              # Double
parameter_3 = [1.0, 2.0, 3.0]   # Array of doubles
```

or more concretely,

```toml
# Example configuration file for frameserve --> framefilt
[frameserve-config]
frame_rate = 30                 # FPS
roi = [10, 10, 100, 100]        # Region of interest

[framefilt-config]
mask = "~/Desktop/mask.png"     # Path to mask file
```

These could then be used in a processing script as follows:

```bash
oat frameserve gige raw -c config.toml frameserve-config &
oat framefilt mask raw filt -c config.toml framefilt-config
```

The type and sanity of parameter values are checked by Oat before they are
used. Below, the type signature, usage information, available configuration
parameters, examples, and configuration options are provided for each Oat
component.

\newpage

### Frame Server
`oat-frameserve` - Serves video streams to shared memory from physical devices
(e.g. webcam or GIGE camera) or from file.

#### Signature
    oat-frameserve --> frame

#### Usage
```
oat-frameserve-help
```

#### Configuration Options
__TYPE = `wcam`__
```
oat-frameserve-wcam-help
```

__TYPE = `gige` and `usb`__
```
oat-frameserve-gige-help
```

__TYPE = `file`__
```
oat-frameserve-file-help
```

__TYPE = `test`__
```
oat-frameserve-test-help
```

#### Examples
```bash
# Serve to the 'wraw' stream from a webcam
oat frameserve wcam wraw

# Stream to the 'graw' stream from a point-grey GIGE camera
# using the gige_config tag from the config.toml file
oat frameserve gige graw -c config.toml gige_config

# Serve to the 'fraw' stream from a previously recorded file
# using the file_config tag from the config.toml file
oat frameserve file fraw -f ./video.mpg -c config.toml file_config
```

\newpage
### Frame Filter
`oat-framefilt` - Receive frames from a frame source, filter, and publish to a
second memory segment. Generally used to pre-process frames prior to object
position detection. For instance, `framefilt` could be used to perform
background subtraction or application of a mask to isolate a region of
interest.

#### Signature
    frame --> oat-framefilt --> frame

#### Usage
```
oat-framefilt-help
```

#### Configuration Options
__TYPE = `bsub`__
```
oat-framefilt-bsub-help
```

__TYPE = `mask`__
```
oat-framefilt-mask-help
```

__TYPE = `mog`__
```
oat-framefilt-mog-help
```

__TYPE = `undistort`__
```
oat-framefilt-undistort-help
```

__TYPE = `thresh`__
```
oat-framefilt-thresh-help
```

#### Examples
```bash
# Receive frames from 'raw' stream
# Perform background subtraction using the first frame as the background
# Publish result to 'sub' stream
oat framefilt bsub raw sub

# Receive frames from 'raw' stream
# Change the underlying pixel color to single-channel GREY
oat framefilt col raw gry -C GREY

# Receive frames from 'raw' stream
# Apply a mask specified in a configuration file
# Publish result to 'roi' stream
oat framefilt mask raw roi -c config.toml mask-config
```

\newpage
### Frame Viewer
`oat-view` - Receive frames from named shared memory and display them on a
monitor. Additionally, allow the user to take snapshots of the currently
displayed frame by pressing <kbd>s</kbd> while the display window is
in focus.

#### Signature
    token --> oat-view

#### Usage
```
oat-view-help
```

#### Configuration Options
__TYPE = `frame`__
```
oat-view-frame-help
```

#### Example
```bash
# View frame stream named raw
oat view frame raw

# View frame stream named raw and specify that snapshots should be saved
# to the Desktop with base name 'snapshot'
oat view frame raw -f ~/Desktop -n snapshot
```

\newpage
### Position Detector
`oat-posidet` - Receive frames from named shared memory and perform object
position detection within a frame stream using one of several methods. Publish
detected positions to a second segment of shared memory.

#### Signature
    frame --> oat-posidet --> position

#### Usage
```
oat-posidet-help
```

#### Configuration Options
__TYPE = `board`__
```
oat-posidet-board-help
```

__TYPE = `diff`__
```
oat-posidet-diff-help
```

__TYPE = `hsv`__
```
oat-posidet-hsv-help
```

__TYPE = `thresh`__
```
oat-posidet-thresh-help
```

#### Example
```bash
# Use color-based object detection on the 'raw' frame stream
# publish the result to the 'cpos' position stream
# Use detector settings supplied by the hsv_config key in config.toml
oat posidet hsv raw cpos -c config.toml hsv_config

# Use motion-based object detection on the 'raw' frame stream
# publish the result to the 'mpos' position stream
oat posidet diff raw mpos
```

\newpage

### Position Generator
`oat-posigen` - Generate positions for testing downstream components.  Publish
generated positions to shared memory.

#### Signature
    oat-posigen --> position

#### Usage
```
oat-posigen-help
```

#### Configuration Options
__TYPE = `rand2D`__
```
oat-posigen-rand2D-help
```

#### Example
```bash
# Publish randomly moving positions to the 'pos' position stream
oat posigen rand2D pos
```

\newpage
### Position Filter
`oat-posifilt` - Receive positions from named shared memory, filter, and
publish to a second memory segment. Can be used to, for example, remove
discontinuities due to noise or discontinuities in position detection with a
Kalman filter or annotate categorical position information based on user supplied
region contours.

#### Signature
    position --> oat-posifilt --> position

#### Usage
```
oat-posifilt-help
```

#### Configuration Options
__TYPE = `kalman`__
```
oat-posifilt-kalman-help
```

__TYPE = `homography`__
```
oat-posifilt-homography-help
```

__TYPE = `region`__
```
oat-posifilt-region-help
```

#### Example
```bash
# Perform Kalman filtering on object position from the 'pos' position stream
# publish the result to the 'kpos' position stream
# Use detector settings supplied by the kalman_config key in config.toml
oat posifilt kalman pos kfilt -c config.toml kalman_config
```

\newpage
### Position Combiner
`oat-posicom` - Combine positions according to a specified operation.

#### Signature
    position 0 --> |
    position 1 --> |
      :            | oat-posicom --> position
    position N --> |

#### Usage
```
oat-posicom-help
```

#### Configuration Options
__TYPE = `mean`__
```
oat-posicom-mean-help
```

#### Example
```bash
# Generate the geometric mean of 'pos1' and 'pos2' streams
# Publish the result to the 'com' stream
oat posicom mean pos1 pos2 com
```

### Frame Decorator
`oat-decorate` - Annotate frames with sample times, dates, and/or positional
information.

#### Signature
         frame --> |
    position 0 --> |
    position 1 --> | oat-decorate --> frame
      :            |
    position N --> |

#### Usage
```
oat-decorate-help
```

#### Example
```bash
# Add textual sample number to each frame from the 'raw' stream
oat decorate raw -s

# Add position markers to each frame from the 'raw' stream to indicate
# objection positions for the 'pos1' and 'pos2' streams
oat decorate raw -p pos1 pos2
```

\newpage
### Recorder
`oat-record` - Save frame and position streams to file.

* `frame` streams are saved as individual video files
* `position` streams are saved as individual [JSON](http://json.org/) files. 

```
{oat-version: X.X},
{header: {timestamp: YYYY-MM-DD-hh-mm-ss},
         {sample_rate_hz: X.X},
         {sources: [ID_1, ID_2, ..., ID_N]} }
{positions: [ [ID_1: position, ID_2: position, ..., ID_N: position ],
              [ID_1: position, ID_2: position, ..., ID_N: position ],

              [ID_1: position, ID_2: position, ..., ID_N: position ] }
}
```
where each `position` object is defined as:

```
{
  samp: Int,                  | Sample number
  unit: Int,                  | Enum spcifying length units (0=pixels, 1=meters)
  pos_ok: Bool,               | Boolean indicating if position is valid
  pos_xy: [Double, Double],   | Position x,y values
  vel_ok: Bool,               | Boolean indicating if velocity is valid
  vel_xy: [Double, Double],   | Velocity x,y values
  head_ok: Bool,              | Boolean indicating if heading  is valid
  head_xy: [Double, Double],  | Heading x,y values
  reg_ok: Bool,               | Boolean indicating if region tag  is valid
  reg: String                 | Region tag
}
```
Data fields are only populated if the values are valid. For instance, in the
case that only object position is valid, and the object velocity, heading, and
region information are not calculated, an example position data point would
look like this:
```
{ samp: 501,
  unit: 0,
  pos_ok: True,
  pos_xy: [300.0, 100.0],
  vel_ok: False,
  head_ok: False,
  reg_ok: False }
```

All streams are saved with a single recorder have the same base file name and
save location (see usage). Of course, multiple recorders can be used in
parallel to (1) parallelize the computational load of video compression, which
tends to be quite intense and (2) save to multiple locations simultaneously.

#### Signature
    position 0 --> |
    position 1 --> |
      :            |
    position N --> |
                   | oat-record
       frame 0 --> |
       frame 1 --> |
         :         |
       frame N --> |

#### Usage
```
oat-record-help
```

#### Example

```bash
# Save positional stream 'pos' to current directory
oat record -p pos

# Save positional stream 'pos1' and 'pos2' to current directory
oat record -p pos1 pos2

# Save positional stream 'pos1' and 'pos2' to Desktop directory and
# prepend the timestamp to the file name
oat record -p pos1 pos2 -d -f ~/Desktop

# Save frame stream 'raw' to current directory
oat record -s raw

# Save frame stream 'raw' and positional stream 'pos' to Desktop
# directory and prepend the timestamp and the word 'test' to each filename
oat record -s raw -p pos -d -f ~/Desktop -n test
```

\newpage
### Position Socket
`oat-posisock` - Stream detected object positions to the network in either
client or server configurations.

#### Signature
    position --> oat-posisock

#### Usage
```
oat-posisock-help
```

#### Configuration Options
__TYPE = `std`__
```
oat-posisock-std-help
```

__TYPE = `pub`__
```
oat-posisock-pub-help
```

__TYPE = `rep`__
```
oat-posisock-rep-help
```

__type = `udp`__
```
oat-posisock-udp-help
```

#### Example
```bash
# Reply to requests for positions from the 'pos' stream to port 5555 using TCP
oat posisock rep pos tcp://*:5555

# Asychronously publish positions from the 'pos' stream to port 5556 using TCP
oat posisock pub pos tcp://*:5556

# Dump positions from the 'pos' stream to stdout
oat posisock std pos
```

\newpage
### Buffer
`oat-buffer` - A first in, first out (FIFO) token buffer that can be use to
decouple asynchronous portions of a data processing network. An example of this
is the case when a precise external clock is used to govern image acquisition
via a physical trigger line. In this case, 'hickups' in the data processing
network following the camera should not cause the camera to skip frames. Of
course, there is no free lunch: if the processing pipline cannot keep up with
the external clock on average, then the buffer will eventually fill and
overflow.

#### Signatures
    position --> oat-buffer --> position

    frame --> oat-buffer --> frame

#### Usage
```
oat-buffer-help
```

#### Example
```bash
# Acquire frames on a gige camera driven by an exnternal trigger
oat frameserve gige raw -c config.toml gige-trig

# Buffer the frames separate asychronous sections of the processing network
oat buffer frame raw buff

# Filter the buffered frames and save
oat framefilt mog buff filt
oat record -f ~/Desktop/ -p buff filt
```

In the above example, one must be careful to fully separate the network across
the buffer boundary in order for it to provide any functionality. For instance,
if we changed the record command to the following
```bash
oat record -f ~/Desktop/ -p raw filt
```
Then the buffer would do nothing since the raw token stream must be synchronous
with the recorder, which bypasses the buffer. In this case, the buffer is just
wasting CPU cycles. Here is a graphical representation of the first
configuration where the `oat-buffer` is used properly. The synchronization
boundary is shown using vertical lines.

```
               |
frameserve --> buffer --> framefilt --> record
               |    \                  /
               |     ------------------
```

In the second configuration, the connection from frameserve to record breaks
the synchronization boundary.
```
               |
frameserve --> buffer --> framefilt --> record
          \    |                      /
           ----|----------------------
```

\newpage

### Calibrate
`oat-calibrate` - Interactive program used to generate calibration parameters
for an imaging system that can be used to parameterize `oat-framefilt` and
`oat-posifilt`. Detailed usage instructions are displayed upon program startup.

#### Signature

    frame --> oat-calibrate

#### Usage
```
oat-calibrate-help
```

#### Configuration Options
__TYPE = `camera`__
```
oat-calibrate-camera-help
```

__TYPE = `homography`__
```
oat-calibrate-homography-help
```

\newpage

### Kill
`oat-kill` - Issue SIGINT to all running Oat processes started by the calling
user. A side effect of Oat's architecture is that components can become
orphaned in certain circumstances: abnormal termination of attached sources or
sinks, running pure sources in the background and forgetting about them, etc.
This utility will gracefully interrupt all currently running oat components.

#### Usage
```
Usage: kill
```

#### Example
```bash
# Interupt all currently running oat components
oat kill
```

\newpage

### Clean
`oat-clean` - Programmer's utility for cleaning shared memory segments after
following abnormal component termination. Not required unless a program
terminates without cleaning up shared memory. If you are using this for things
other than development, then please submit a bug report.

#### Usage
```
oat-clean-help
```

#### Example
```bash
# Remove raw and filt blocks from shared memory after abnormal terminatiot of
# some components that created them
oat clean raw filt
```

\newpage

## Installation
First, ensure that you have installed all dependencies required for the
components and build configuration you are interested in in using. For more
information on dependencies, see the [dependencies](#dependencies) section
below. To compile and install Oat, starting in the top project directory,
create a build directory, navigate to it, and run cmake on the top-level
CMakeLists.txt like so:

```bash
mkdir release
cd release
cmake -DCMAKE_BUILD_TYPE=Release [CMAKE OPTIONS] ..
make
make install
```

If you just want to build a single component component, individual components
can be built using `make [component-name]`, e.g. `make oat-view`. Available
cmake options and their default values are:

```
-DUSE_FLYCAP=Off // Compile with support for Point Grey Cameras
-DBUILD_DOCS=Off     // Generate Doxygen documentation
```

If you had to install Boost from source, you must let cmake know where it is
installed via the following switch. Obviously, provide the correct path to the
installation on your system.

```
-DBOOST_ROOT=/opt/boost_1_59_0
```

To complete installation, add the following to your `.bashrc` or equivalent.
This makes Oat commands available within your user profile (once you start a new terminal):

```bash
# Make Oat commands available to user
eval "$(<path/to/Oat>/oat/bin/oat init -)"
```

If you get runtime link errors when you try to run an Oat program such as
>error while loading shared libraries: libboost_program_options.so.1.60.0
then you need to ad the following entry to your `.bashrc`

```bash
export LD_LIBRARY_PATH=</path/to/boost>/stage/lib:$LD_LIBRARY_PATH
```

### Dependencies
#### License compatibility

Oat is licensed under the
[GPLv3.0](http://choosealicense.com/licenses/gpl-3.0/). Its dependences' are
licenses are shown below:

- Flycapture SDK: NON-FREE specialized license (This is an optional package. If
  you compile without Flycapture support, you can get around this. Also, see
  the `GigE interface cleanup` entry in the TODO section for a potentially free
  alternative.)
- OpenCV: BSD
- ZeroMQ: LGPLv3.0
- Boost: Boost software license
- cpptoml: Some kind of Public Domain Dedication
- RapidJSON: BSD
- Catch: Boost software license

These licenses do not violate the terms of Oat's license. If you feel otherwise
please submit an bug report.

#### Flycapture SDK
The FlyCapture SDK is used to communicate with Point Grey digital cameras. It
is not required to compile any Oat components.  However, the Flycapture SDK is
required if a Point Grey camera is to be to be used with the `oat-frameserve`
component to acquire images. If you simply want to process pre-recorded files
or use a web cam, e.g. via

```
oat-frameserve file raw -f video.mpg
oat-frameserve wcam raw
```

then this library is _not_ required.

To install the Point Grey SDK:

- Go to [point-grey website](www.ptgrey.com)
- Download the FlyCapture2 SDK (version >= 2.7.3). Annoyingly, this requires you
  to create an account with Point Grey.
- Extract the archive and use the `install_flycapture.sh` script to install the
  SDK on your computer and run

```bash
tar xf flycapture.tar.gz
cd flycapture
sudo ./install_flycapture
```

#### Boost
The [Boost libraries](http://www.boost.org/) are required to compile all Oat
components. You will need to install versions >= 1.56. To
install Boost, use APT or equivalent,

```bash
sudo apt-get install libboost-all-dev
```

If you are using an Ubuntu distribution older than Wily Werewolf, Boost will be
too old and you will need to install from source via

```bash
# Install latest boost
wget http://sourceforge.net/projects/boost/files/latest/download?source=files -O tarboost
tar -xf tarboost
cd ./boost*
./bootstrap.sh
./b2 --with-program_options --with-system --with-thread --with-filesystem
cd ..
sudo mv boost* /opt
```

Finally, if you are getting runtime linking errors, you will need to place the
following in `.bashrc`
```bash
export LD_LIBRARY_PATH=<path to boost root directory>/stage/lib:$LD_LIBRARY_PATH
```

#### OpenCV
[opencv](http://opencv.org/) is required to compile the following oat components:

- `oat-frameserve`
- `oat-framefilt`
- `oat-view`
- `oat-record`
- `oat-posidet`
- `oat-posifilt`
- `oat-decorate`
- `oat-positest`

__Note__: OpenCV must be installed with ffmpeg support in order for offline
analysis of pre-recorded videos to occur at arbitrary frame rates. If it is
not, gstreamer will be used to serve from video files at the rate the files
were recorded. No cmake flags are required to configure the build to use
ffmpeg. OpenCV will be built with ffmpeg support if something like

```
-- FFMPEG:          YES
-- codec:           YES (ver 54.35.0)
-- format:          YES (ver 54.20.4)
-- util:            YES (ver 52.3.0)
-- swscale:         YES (ver 2.1.1)
```

appears in the cmake output text. The dependencies required to compile OpenCV
with ffmpeg support, can be obtained as follows:

```bash
TODO
```

__Note__: To increase Oat's video visualization performance using `oat view`,
you can build OpenCV with OpenGL and/or OpenCL support. Both will open up
significant processing bandwidth to other Oat components and make for faster
processing pipelines. To compile OpenCV with OpenGL and OpenCL support, first
install dependencies:

```
sudo apt-get install libgtkglext1 libgtkglext1-dev
```

Then, add the `-DWITH_OPENGL=ON` and the `-DWITH_OPENCL=ON` flags to the cmake
command below.  OpenCV will be build with OpenGL and OpenCL support if `OpenGL
support: YES` and `Use OpenCL: YES` appear in the cmake output text. If OpenCV
is compiled with OpenCL and OpenGL support, the performance benefits will be
automatic, no compiler options need to be set for Oat.

__Note__: If you have [NVIDIA GPU that supports
CUDA](https://developer.nvidia.com/cuda-gpus), you can build OpenCV with CUDA
support to enable GPU accelerated video processing.  To do this, will first
need to install the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit).

- Be sure to __carefully__ read the [installation
  instructions](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/index.html)
  since it is a multistep process. Here are some additional hints that worked
  for me:
- I have found that installing the toolkit via 'runfile' to be the most
  painless. To do this you will need to switch your system to text mode using
  <kbd>Ctrl</kbd> + <kbd>Alt</kbd> + <kbd>F1</kbd>, and killing the X-server
  via `sudo service lightdm stop` (or equivalent), and running the runfile with
  root privileges.
- I have had the most success on systems that do not use GNOME or other fancy
  desktop environments. The install on [lubunut](http://lubuntu.net/), which
  uses LXDE as its desktop environment, was especially smooth.
- Do __not__ install the nvidia drivers along with the CUDA toolkit
  installation. I found that (using ubuntu 14.04) this causes all sorts of
  issues with X, cinnamon, etc, to the point where I could not even boot my
  computer into anything but text mode. Instead, install the NVIDIA drivers
  using either the package manager (`nvidia-current`) or even more preferably,
  using the [`device-drivers`]'(http://askubuntu.com/a/476659) program or
  equivalent.
- If you hare getting a `cv::exception` complaining that about
  `code=30(cudaErrorUnknown) "cudaGetDeviceCount(&device_count)"` or similar,
  run the affected command as [root one
  time](https://devtalk.nvidia.com/default/topic/699610/linux/334-21-driver-returns-999-on-cuinit-cuda-/).

If OpenCV is compiled with CUDA suport, the CUDA-enabled portions of the Oat
codebase will be enabled automatically. No compile flags are required.

__Note__: GUI functionality is enhanced in OpenCV is compiled with Qt support.
You can build OpenCV with Qt by first installing the [Qt
SDK](http://download.qt.io/official_releases/online_installers/qt-unified-linux-x64-online.run)
and these dependencies:

```
# Additional dependencies for integraged QT with OpenGL
sudo apt-get install libqt5opengl5 libqt5opengl5-dev
```

The you can compile OpenCV using QT support by adding `-DWITH_QT=ON` flag to
the cmake command below. QT functionality will then be used by Oat
automatically.

Finally, to compile and install OpenCV:

```bash
# Install OpenCV's dependencies
sudo apt-get install build-essential # Compiler
sudo apt-get install cmake git # For building opencv and Oat
sudo apt-get install libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libv4l-dev # Required
sudo apt-get install libv4l-dev # Allows changing frame rate with webcams
sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
sudo apt-get install # ffmpeg support [TODO]
sudo apt-get install # OpenGL support [TODO]
sudo ldconfig -v

# Get OpenCV
wget https://github.com/Itseez/opencv/archive/3.1.0.zip -O opencv.zip
unzip opencv.zip -d opencv

# Build OpenCV
cd opencv/opencv-3.0.0-rc1
mkdir release
cd release

# Run cmake to generate Makefile
# Add -DWITH_CUDA=ON for CUDA support and -DWITH_OPENGL for OpenGL support
cmake -DWITH_LIBV4L=ON -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local ..

# Build the project and install
make
sudo make install
```

#### ZeroMQ
[ZeroMQ](zeromq.org) is required by the following
Oat components:

- `oat-record`
- `oat-posisock`

Download, compile, and install ZeroMQ as follows:

```bash
wget http://download.zeromq.org/zeromq-4.1.4.tar.gz -O tarzmq
tar -xf tarzmq
cd ./zeromq*
./configure --without-libsodium
make
sudo make install
sudo ldconfig
```

Additionally, you will need to download the ZeroMQ C++ binding (this is just a
single header file) and place it somewhere that your compiler will find it.

```bash
wget https://raw.githubusercontent.com/zeromq/cppzmq/master/zmq.hpp
sudo mv zmq.hpp /usr/local/include/
```

#### RapidJSON, cpptoml, and Catch
These libraries are installed automatically by cmake during the build process.

[RapidJSON](https://github.com/miloyip/rapidjson) is required by the following
Oat components:

- `oat-record`
- `oat-posisock`

[cpptoml](https://github.com/skystrife/cpptoml) is required by the following
Oat components:

- `oat-frameserve`
- `oat-framefilt`
- `oat-posidet`
- `oat-posifilt`
- `oat-posicom`
- `oat-positest`

[Catch](https://github.com/philsquared/Catch) is required to make and run tests
using `make test`

## Performance
Oat is designed for use in real-time video processing scenarios. This boils
down the following definition

> The average execution time for an Oat dataflow network must not exceed the
> camera(s) image transfer period

If this condition is not met, then frames will eventually be dropped. There is
no way around this. The guts of Oat consist of a simple, but very efficient
[message passing library](/lib/shmemdf) that links together processing
routines taken from a variety of sources (some written by me, some by third
party projects such as OpenCV). The speed of each processing step is determined
both by its computational complexity and deftness of implementation, both of
which can vary quite a lot for different components. To see some rudimentary
performance numbers for Oat components in isolation, have a look at [these
numbers](test/perf/results.md). There is definitely room for optimization for
some components. And, several components that are ripe for GPU implementation
do not have one yet. This comes down to free time. If anyone wants to try there
hand at making some of the bottleneck components faster, please get in touch.

Outside of code optimization, there are a few things a user should be aware of
to make efficient use of Oat, which are listed below.

### Frames are slow
The first thing to know is that working with `frames` is orders of magnitude
slower than working with `positions`. Therefore, minimizing the number of
processing steps operating on `frames` is a good way to reduce computational
requirements. Processing on `positions` is in the noise in comparison.

### Parallelism
Increasing the number of components in your chain does not necessarily cause an
appreciable an increase in processing time because Oat components run in
parallel.  Instead, up to the limit of the number of hyperthreads/GPU resources
your computer supports, the slowest component in a dataflow network will
largely determine the speed of the processing rather than the number
of components within the processing network.

### Resolution
Do you really need that 10 MP camera? Recall that increases in sensor
resolution cause a power 2 increase in then number of pixels you need to smash
into RAM, process, write to disk, and, probably, post process. Its really best
to use the lowest resolution camera that suites your needs, both for the sake
of real-time processing in Oat and your future sanity when trying to deal with
those 30 GB video files.

### Hard-disk
If you are saving video, then the write speed of your hard disk can become the
limiting factor in a processing network. To elaborate, I'm just quoting my
response to [this issue](https://github.com/jonnew/Oat/issues/14):

> __Q:__ I also ran into an issue with RAM and encoding. I have 8 GB, and they fill up
> within about 20 seconds, then goes into swap.

> __A:__ I suspect the following is the issue:
>
> (22 FPS * 5 MP * 24 bits/pixel) / ( 8 bits/ byte) = 330 MB/sec
>
> This (minus compression, which I'm admittedly ignoring, but is probably made
> up for by the time it takes to do the compression...) is the requisite write
> speed (in actuality, not theoretically) of your hard disk in order not to get
> memory overflow.
>
> 8 GB / 0.330 GB =~ 24 seconds.
>
> The RAM is filling because your hard disk writes are not occurring fast
> enough. Oat is pushing frames to be written into a FIFO in main memory that
> the recorder thread is desperately trying to write to disk. Getting more
> RAM will just make the process persist for a bit longer before failing. I
> would get an SSD for streaming video to and then transfer those videos to
> a slower long term storage after recording.

##  Setting up a Point-grey PGE camera in Linux
`oat-frameserve` supports using Point Grey GIGE cameras to collect frames. I
found the setup process to be straightforward and robust, but only after
cobbling together the following notes.

### Camera IP Address Configuration
First, assign your camera a static IP address. The easiest way to do this is to
use a Windows machine to run the Point Grey 'GigE Configurator'. If someone
knows a way to do this without Windows, please tell me. An example IP
Configuration might be:

- Camera IP: 192.168.0.1
- Subnet mask: 255.255.255.0
- Default gateway: 192.168.0.64

### Point Grey GigE Host Adapter Card Configuration
Using network manager or something similar, you must configure the IPv4
configuration of the GigE host adapter card you are using to interface the
camera with your computer.

- First, set the ipv4 method to __manual__.
- Next, you must configure the interface to (1) have the same network prefix
  and (2) be on the same subnet as the camera you setup in the previous
  section.
    - Assuming you used the camera IP configuration specified above, your host
      adapter card should be assigned the following private IPv4 configuration:
          - POE gigabit card IP: 192.168.0.100
          - Subnet mask: 255.255.255.0
          - DNS server IP: 192.168.0.1
- Next, you must enable jumbo frames on the network interface. Assuming that
  the camera is using `eth2`, then entering

        sudo ifconfig eth2 mtu 9000

  into the terminal will enable 9000 MB frames for the `eth2` adapter.
- Finally, to prevent image tearing, you should increase the amount of memory
  Linux uses for network receive buffers using the `sysctl` interface by typing

        sudo sysctl -w net.core.rmem_max=1048576 net.core.rmem_default=1048576

  into the terminal. _In order for these changes to persist after system
  reboots, the following lines must be added to the bottom of the
  `/etc/sysctl.conf` file_:

        net.core.rmem_max=1048576
        net.core.rmem_default=1048576

  These settings can then be reloaded after reboot using

        sudo sysctl -p

### Multiple Cameras
- If you have two or more cameras/host adapter cards,  they can be configured
  as above but _must exist on a separate subnets_. For instance, we could
  repeat the above configuration steps for a second camera/host adapter card
  using the following settings:
    - Camera Configuration:
          - Camera IP: 192.168.__1__.1
          - Subnet mask: 255.255.255.0
          - Default gateway: 192.168.__1__.64
    - Host adapter configuration:
          - POE gigabit card IP: 192.168.__1__.100
          - Subnet mask: 255.255.255.0
          - DNS server IP: 192.168.__1__.1

### Example Camera Configuration
Below is an example network adapter and camera configuration for a two-camera
imaging system provided by [Point Grey](http://www.ptgrey.com/). It consists of
two Blackfly GigE cameras (Point Grey part number: BFLY-PGE-09S2C) and a single
dual-port POE GigE adapter card (Point Grey part number: GIGE-PCIE2-2P02).

#### Camera 0

- Adapter physical connection (looking at back of computer)
```
RJ45 ------------
          |      |
    L [  [ ]    [x]  ] R
```

- Adapter Settings
    - Model:         Intel 82574L Gigabit Network Connection
    - MAC:           00:B0:9D:DB:D9:63
    - MTU:           9000
    - DHCP:          Disabled
    - IP:            192.168.0.100
    - Subnet mask:   255.255.255.0
- Camera Settings
    - Model:         Blackfly BFLY-PGE-09S2C
    - Serial No.:    14395177
    - IP:            192.168.0.1 (Static)
    - Subnet mask:   255.255.255.0
    - Default GW:    0.0.0.0
    - Persistent IP: Yes

#### Camera 1

- Adapter physical connection (looking at back of computer)
```
RJ45 ------------
          |      |
    L [  [x]    [ ]  ] R
```

- Adapter Settings
    - Model:         Intel 82574L Gigabit Network Connection
    - MAC:           00:B0:9D:DB:A7:29
    - MTU:           9000
    - DHCP:          Disabled
    - IP:            192.168.1.100
    - Subnet mask:   255.255.255.0
- Camera Settings
    - Model:         Blackfly BFLY-PGE-09S2C
    - Serial No.:
    - IP:            192.168.1.1 (Static)
    - Subnet mask:   255.255.255.0
    - Default GW:    0.0.0.0
    - Persistent IP: Yes

\newpage

## TODO
- [ ] Unit and stress testing
    - Unit tests for `libshmemdf`
        - ~~Nominal data types, `T`~~
        - Specializations for `Frames`
    - Stress tests for data processing chains
        - I need to come up with a series of scripts that configure and run
          components in odd and intensive, but legal, ways to ensure sample
          sychronization is maintained, graceful exits, etc
- [ ] Position type correction
    - It might be a good idea to generalize the concept of a position to a
      multi-positional element
    - For things like the `oat-decorate`, `oat-posicom`, and potentially
      `oat-detect`, this could increase performance and decrease user script
      complexity if multiple targets common detection features needed to be
      tracked at once.
    - Down side is that it potentially increases code complexity and would
      require a significant refactor.
    - Additionally, position detection might no longer be stateless. E.g. think
      of the case when two detected objects cross paths. In order to ID the
      objects correctly in subsequent detections, the path of the objects would
      need to be taken into account (and there is not guarantee this result
      will be correct...). A potential work around is to have IDed 'position
      groups' with annoymous position members. This would get us back to
      stateless detection. However, it would make the concept of position
      combining hard to define (although that is even true now is just a design
      choice, really).
    - EDIT: Additionally, there should certainly not be `Position2D` vs
      `Position3D`. Only `Position` which provides 3d specificaiton with Z axis
      defaulting to 0.
- [ ] [CBOR](http://tools.ietf.org/html/rfc7049) binary messaging and data
  files
    - CBOR is a simple binary encoding scheme for JSON
    - It would be great to allow the option to save CBOR files (`oat-record`)
      or send CBOR messages (`oat-posisock`) by creating a CBOR `Writer`
      acceptable to by `Position` datatype's serialization function.
    - And, while I'm at it, Position's should be forced to support
      serialization, so this should be a pure abstract member of the base
      class.
    - Another option that is very similar is messagepack. Don't know which is
      better.
- [ ] `oat-framefilt undistort`
    - Very slow. Needs an OpenGL or CUDA implementation
    - User supplied frame rotation occurs in a separate step from
      un-distortion.  Very inefficient. Should be able to combine rotation with
      camera matrix to make this a lot faster.
- [ ] Should components always involve a user IO thread?
    - For instance, some generalization of `oat-record ... --interactive`
    - For instance, it would be nice if PURE SINKs (e.g. `oat frameserve`)
      could have their sample clock reset via user input, without having to
      restart the program.
    - For instance, it would be nice to be able to re-acquire the background
      image in `oat-framefilt bsub` without have to restart the program.
    - Where should this come from? Command line input?
- [ ] Add position history toggle in `oat-decorate`
- [ ] Type deduction in shmem Tokens
    - Sources should have a static method for checking the token type of a
      given address.
