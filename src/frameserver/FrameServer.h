//******************************************************************************
//* File:   FrameServer.h
//* Author: Jon Newman <jpnewman snail mit dot edu>
//*
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu) 
//* All right reserved.
//* This file is part of the Oat project.
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
//*****************************************************************************

#ifndef FRAMESERVER_H
#define	FRAMESERVER_H

#include <atomic>
#include <opencv2/opencv.hpp>

#include "../../lib/shmem/SharedMemoryManager.h"
#include "../../lib/shmem/BufferedMatServer.h"

/**
 * Abstract base class to be implemented by any Camera Server within the Simple
 * Tracker project.
 * @param image_sink_name Image SINK name.
 */
class FrameServer {
public:
    
    FrameServer(std::string image_sink_name) : 
      name("frameserve[" + image_sink_name + "]")
    , frame_sink(image_sink_name)
    , undistort_image(false)
    , current_sample(0) { }
    
    /**
     * Cameras must be able to serve cv::Mat frames.
     * @return running state. true = stream EOF (e.g. at end of file). false = stream not exhausted.
     */
    virtual bool serveFrame(void) {
        
        grabFrame(current_frame);
        undistortFrame(); // TODO: move to frame filt
        
        if (!current_frame.empty()) {
            
            frame_sink.pushMat(current_frame, current_sample);
            current_sample++;
            
            return false;
        } else {
            
            stop();
            return true;
        }
    };
    
    // Cameras allow image undistortion if parameters are provided
    // TODO: This should absolutely be a framefilt component
    void undistortFrame(void) {
        if (undistort_image) {
            cv::Mat undistorted_frame;
            cv::undistort(current_frame, undistorted_frame, camera_matrix, distortion_coefficients);
            current_frame = undistorted_frame;
        }
    }
 
    // Cameras must be configurable via file
    //virtual void configure(void) = 0;
    virtual void configure() = 0;
    virtual void configure(const std::string& file_name, const std::string& key) = 0;
    
    cv::Mat get_current_frame(void) const { return current_frame; }
    virtual std::string get_name(void) const { return name; }
    
    // Cameras must be interruptable by the user in a way that ensures shmem
    // is freed
    void stop(void) { frame_sink.set_running(false); }

protected:
    
    // Cameras must be able to obtain a cv::Mat from some source (physical camera, file, etc)
    virtual void grabFrame(cv::Mat& frame) = 0;
    
    // Server name
    std::string name;
    
    // Cameras have a region of interest to crop images
    cv::Rect region_of_interest;
    
    // TODO: Move to framefilt
    // Camera matrix and distortion coefficients. Use to undistort image
    bool undistort_image;
    cv::Mat camera_matrix; // TODO: change to Matx
    cv::Mat distortion_coefficients; // TODO: change to Matx
   
private:
    
    
    // cv::Mat server for sending frames to shared memory
    oat::BufferedMatServer frame_sink;
    
    // Currently acquired frame
    cv::Mat current_frame;
    
    // Current sample number  ( this does account for missed hardware triggers)
    uint32_t current_sample;

};

#endif	/* FRAMESERVER_H */

