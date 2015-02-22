Simple Driver for Point Grey Cameras
=======

This software configures a point grey cameras such that digital pulses on a user selected GPIO line yield a single frame. 

# Notes on project features and scope

## Stage 1 - Investigate camera configuration
- What are the options for triggered capture, and what API call to use in the triggered case (perhaps the same as the free running API call?). 
  - The options are given in a set of registers that are set via USB.
- Or is there a hardware line that tells camera triggered vs. Free running?
  - No, the options are set programatically. 
  - The trigger is a hardware line though and should be governed by the master recording clock (__not__ by software).


## Stage 2 - Pass grabbed frames to the tracker
- Requested features
- Server has enough feature parity to not be painful compared to direct library access: 
  - Direct access to what library? To flycapture? 
  - Flycapture has lots of features that are irrelevant for the task at hand. I thought we want to bypass FC as much as possible and get standardized data to something more capable of processing it.
- Wire format: per packet, one time-stamp and N frames labeled by camera serial number. Frames encoded to something like rgb8 char array
- Multiple clients
  - Basically any widely used IPC route will support this
- Some amount of RPC. Don't want to be serving the output while forcing users to locally admin yet another process
  - I'm not sure I understand this, except maybe in the context of starting or stopping the server
- Strongly prefer to consume JSON over something ad hoc, opaque and untyped
  - There will need to be some encoding/decoding steps if we use JSON, which has not native support for binary data blocks.
  - Using an JSON array of JSON numbers or JSON strings to represent RGB values will result in unreasonably inefficient encoding of data, and packing and parsing will be slow.
  - Using a Base64 scheme, we can trick a JSON string into holding a binary data block representing the image. It will still be a named property of the object.

# TODO
- [ ] Frame format?
  - See above discussion on JSON   
- [ ] Frame serialization?
  - JSON with Base64-encoded strings holding binary rgb8 arrays
- [ ] Frame buffer?
  - Probably handled by IPC method if chosen correctly
- [ ] IPC method?
  - UPD, TCP, shared memory, pipe?
- [ ] RPC infrastructure
  - Which methods will the client need access to?

