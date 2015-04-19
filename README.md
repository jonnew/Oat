##Simple soft real-time position tracker for Point Grey Cameras

- Example applications using the flycapture SDK are available in /usr/src/flycapture/src

### TODO
- [ ] Configuration class should specify frame capture due to digital pulses on a user selected GPIO line or free running.
- [ ] Interprocess data processing synchronization
  - Whatever is chosen, all subsequent processing must propagate in accordance with the frame captured by the base image server(s).
  - e.g.

        Camera --> Background Subtract --> Detector --> Decorator --> Viewer
               ╲                                      ╱          ╲
	             ------------------------------------              -> Recorder    	
     In this case, Decorator must block until Detector provides a result. However, Camera _may_ have produce another image in the meantime causing the Detector result and the image used by the Decorator to be out of sync. I need to find an intellegent way to deal with this.
- [ ] Start synchronization. The data processing chain should be initialized and waiting before the first image get sent.
- [ ] Frame buffer?
- [x] IPC method?
  - ~~UPD, TCP,~~ **shared memory**, ~~pipe?~~
- [ ] Networked communication with clients that use extracted positional information
- [ ] General C++ coding practice
  - Pass by const ref whenever possible. Especially relevant when passing derived objects to prevent slicing.
  - const member properties can be initialized in the initialization list, rather than assigned in the constructor body. Take advantage.
- [ ] Programs to implment
  - Pure intensity based detector (now color conversion, just saturation on raw image)
  - Position Filter (Kalman is first implementation)
  - Recorder (Position and images? Viewer can also record?)



### Passing positional data to the client process 

#### Ideas...
- Wire format: per packet, one time-stamp and N frames labeled by camera serial number. Frames encoded to something like rgb8 char array
  - Strongly prefer to consume JSON over something ad hoc, opaque and untyped
  - There will need to be some encoding/decoding steps if we use JSON, which has not native support for binary data blocks.
  - Using an JSON array of JSON numbers or JSON strings to represent RGB values will result in unreasonably inefficient encoding of data, and packing and parsing will be slow.
  - Using a Base64 scheme, we can trick a JSON string into holding a binary data block representing the image. It will still be a named property of the object.
- Multiple clients
  - Broadcast over UDP
  - Shared memory (no good for remote tracker)
  - TCP/IP with thread for each client 

### Connecting to point-grey PGE camera in Linux

- First you must assign your camera a static IP address. The easiest way to do this is to use a Windows machine to run the 
- The ipv4 method should be set to manual.
- Finally, you must the PG POE gigabit interface to (1) have the same network prefix and (2) be on the same subnet as your Gigabit camera. For instance, assume that your camera was assigned the following private ipv4 configuration:
  - Camera IP: 192.168.0.1
  - Subnet mask: 255.255.255.0
  In this case, a functional ipv4 configuration for the POE Gigabit Ethernet card in the host PC could be:
  - POE gigabit card IP: 192.168.0.100
  - Subnet mask: 255.255.255.0
- Not that if you want to add another network interface for another camera, it must exist on a separate subnet! For instance, we could repeat the above configuration steps using the following settings:
  - Camera IP: 192.168.1.1
  - Subnet mask: 255.255.255.0
In this case, a functional ipv4 configuration for the POE Gigabit Ethernet card in the host PC could be:
  - POE gigabit card IP: 192.168.1.100
  - Subnet mask: 255.255.255.0
