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

#ifndef CAMERA_H
#define	CAMERA_H

#include "../../lib/shmem/MatServer.h"

/**
 * Abstract base class to be implemented by any Camera Server within the Simple
 * Tracker project.
 * @param image_sink_name Image SINK name.
 */
class Camera {
public:
    
    Camera(std::string image_sink_name) : 
      name(image_sink_name)
    , frame_sink(image_sink_name) { }
    
    // Cameras must be able to serve cv::Mat frames
    virtual void serveMat(void) = 0;
    
    // Cameras must be able to obtain a cv::Mat from some source (physical camera, file, etc)
    virtual void grabMat(void) = 0;
    
    // Cameras must be configurable via file
    virtual void configure(void) = 0;
    virtual void configure(std::string file_name, std::string key) = 0;
    
    // Cameras must be interuptable
    void stop(void) { frame_sink.set_running(false); frame_sink.notifySelf(); };
    
protected:
    
    // cv::Mat server for sending frames to shared memory
    MatServer frame_sink;
    std::string name;
    
};

#endif	/* CAMERA_H */

