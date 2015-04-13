#include "WebCam.h"


WebCam::WebCam(std::string name) :
  camera_name(name)
, camera(0) {}

WebCam::serveMat() {

	camera >> current_frame;
	frame_sink.set_shared_mat(current_frame);
}
