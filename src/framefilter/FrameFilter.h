//******************************************************************************
//* File:   FrameFilt.h
//* Author: Jon Newman <jpnewman snail mit dot edu>
//
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu) 
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

#ifndef FRAMEFILT_H
#define	FRAMEFILT_H

#include <string>
#include <opencv2/core/mat.hpp>

#include "../../lib/shmem/SharedMemoryManager.h"
#include "../../lib/shmem/MatClient.h"
#include "../../lib/shmem/MatServer.h"

class FrameFilter {
public:

    FrameFilter(const std::string& source_name, const std::string& sink_name) :
      name("framefilt[" + source_name + "->" + sink_name + "]")
    , frame_source(source_name)
    , frame_sink(sink_name) { }
    
    virtual ~FrameFilter() { }

    // Frame filters must be able to receive, filter, and serve frames
    bool processSample(void) {
        
        // Only proceed with processing if we are getting a valid frame
        if (frame_source.getSharedMat(current_frame)) {

            // Push filtered frame forward, along with frame_source sample number
            frame_sink.pushMat(filter(current_frame), frame_source.get_current_sample_number());
        }

        return (frame_source.getSourceRunState() == oat::ServerRunState::END);
    }

    // Frame filters must be configurable
    virtual void configure(const std::string& config_file, const std::string& config_key) = 0;
    
    // Frame filters have a descriptive, accessible name
    std::string get_name(void) const { return name; }
    
protected:
    
    virtual cv::Mat filter(cv::Mat& input_frame) = 0;

private:

    // Component name
    std::string name;
    
    // Frame filters have Mat client object for receiving frames
    cv::Mat current_frame;
    
    // Frame filters have a frame source from which frames are received
    oat::MatClient frame_source;

    // Frame filters have Mat server for sending processed frames
    oat::MatServer frame_sink;
   
};

#endif	/* FRAMEFILT_H */

