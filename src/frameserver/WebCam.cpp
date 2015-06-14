//******************************************************************************
//* Copyright (c) Jon Newman (jpnewman at mit snail edu) 
//* All right reserved.
//* This file is part of the Simple Tracker project.
//* This is free software: you can redistribute it and/or modify
//* it under the terms of the GNU General Public License as published by
//* the Free Software Foundation, either version 3 of the License, or
//* (at your option) any later version.
//* This software is distributed in the hope that it will be useful,
//* but WITHOUT ANY WARRANTY; without even the implied warranty of
//* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//* GNU General Public License for more details.
//* You should have received a copy of the GNU General Public License
//* along with this source code.  If not, see <http://www.gnu.org/licenses/>.
//******************************************************************************

#include "WebCam.h"

#include <string>

WebCam::WebCam(std::string frame_sink_name) :
  Camera(frame_sink_name)
, cv_camera(0) {
}

void WebCam::grabMat() {
    cv_camera >> current_frame;
}

bool WebCam::serveMat() {
    frame_sink.pushMat(current_frame, current_sample++);
    return false;
}

void WebCam::configure() { }
void WebCam::configure(std::string file_name, std::string key) { }