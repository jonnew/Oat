#include "WebCam.h"

WebCam::WebCam(std::string name) :
  camera_name(name)
, frame_sink(name) 
, camera(0) {
}

void WebCam::serveMat() {

    camera >> current_frame;
    frame_sink.set_shared_mat(current_frame);
}
