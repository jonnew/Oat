Simple soft real-time position tracker for Point Grey Cameras
=======

- Example applications using the flycapture SDK are available in /usr/src/flycapture/src

# TODO
- [ ] IPC with clients who use extracted positional information
- [ ] Configuration class should specify frame capture due to digital pulses on a user selected GPIO line. 

# Notes on project features and scope

## Stage 1 - Investigate camera configuration
- What are the options for triggered capture, and what API call to use in the triggered case (perhaps the same as the free running API call?). 
  - The options are given in a set of registers that are set via USB.
- Or is there a hardware line that tells camera triggered vs. Free running?
  - No, the options are set programatically. 
  - The trigger is a hardware line though and should be governed by the master recording clock (__not__ by software).

## Stage 2 - Pass grabbed frames to the tracker
- Wire format: per packet, one time-stamp and N frames labeled by camera serial number. Frames encoded to something like rgb8 char array
  - Strongly prefer to consume JSON over something ad hoc, opaque and untyped
  - There will need to be some encoding/decoding steps if we use JSON, which has not native support for binary data blocks.
  - Using an JSON array of JSON numbers or JSON strings to represent RGB values will result in unreasonably inefficient encoding of data, and packing and parsing will be slow.
  - Using a Base64 scheme, we can trick a JSON string into holding a binary data block representing the image. It will still be a named property of the object.
- Multiple clients
  - Broadcase over UDP
  - Shared memory (no good for remote tracker)
  - TCP/IP with thread for each client 

## Stage 3 - Tracker processes frames
- Use OpenCV
  - See [this tutorial](https://youtu.be/bSeFrPrqZ2A?list=PLvwB65U8V0HHCEyW2UTyOJym5FsdqfbHQ)
- Processed data packet
  - Another JSON object?
  - 
# TODO
- [ ] Frame format?
  - See above discussion on JSON   
- [ ] Frame serialization?
  - JSON with Base64-encoded strings holding binary rgb8 arrays
- [ ] Frame buffer?
  - Probably handled by IPC method if chosen correctly
- [ ] IPC method?
  - UPD, TCP, shared memory, pipe?

