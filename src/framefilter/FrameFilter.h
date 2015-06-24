//******************************************************************************
//* File:   FrameFilter.h
//* Author: Jon Newman <jpnewman snail mit dot edu>
//*
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

    /**
     * Abstract frame filter.
     * All concrete frame filter types derive from this ABC.
     */
    FrameFilter(const std::string& source_name, const std::string& sink_name) :
      name("framefilt[" + source_name + "->" + sink_name + "]")
    , frame_source(source_name)
    , frame_sink(sink_name) { }
    
    virtual ~FrameFilter() { }

    /**
     * Obtain raw frame from SOURCE. Apply filter function to raw frame. Publish
     * filtered frame to SINK.
     * @return SOURCE end of stream signal. If true, this component should exit.
     */
    bool processSample(void) {
        
        // Only proceed with processing if we are getting a valid frame
        if (frame_source.getSharedMat(current_frame)) {

            // Push filtered frame forward, along with frame_source sample number
            frame_sink.pushMat(filter(current_frame), frame_source.get_current_sample_number());
        }

        return (frame_source.getSourceRunState() == oat::ServerRunState::END);
    }

    /**
     * Configure filter parameters.
     * @param config_file configuration file path
     * @param config_key configuration key
     */
    virtual void configure(const std::string& config_file, const std::string& config_key) = 0;
    
    /**
     * Get FrameFilter name
     * @return name 
     */
    std::string get_name(void) const { return name; }
    
protected:
    
    /**
     * Perform frame filtering.
     * @param frame unfiltered frame
     * @return filtered frame
     */
    virtual cv::Mat filter(cv::Mat& frame) = 0;

private:

    // Filter name.
    std::string name;
    
    //Currently processed frame
    cv::Mat current_frame;
    
    // Frame SOURCE object for receiving raw frames
    oat::MatClient frame_source;
    
    //Frame SOURCE object for publishing filtered frames
    oat::MatServer frame_sink;
};

#endif	/* FRAMEFILT_H */

