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

#ifndef WEBCAM_H
#define WEBCAM_H

#include <string>

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp> // TODO: correct header...

#include "Camera.h"
#include "../../lib/shmem/SharedCVMatHeader.h"
#include "../../lib/shmem/MatServer.h"

class WebCam : public Camera {
public:
    WebCam(std::string frame_sink_name);

    // Implement Camera interface
    void configure(void); 
    void configure(std::string config_file, std::string key);
    void grabMat(void);
    bool serveMat(void);

private:
    
    bool aquisition_started;

    // The webcam object
    cv::VideoCapture cv_camera;

};
#endif //WEBCAM_H
