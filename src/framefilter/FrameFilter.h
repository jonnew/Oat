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

#ifdef OAT_USE_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#endif

#include "../../lib/shmem/SharedMemoryManager.h"
#include "../../lib/shmem/MatClient.h"
#include "../../lib/shmem/MatServer.h"

/**
 * Abstract frame filter.
 * All concrete frame filter types implement this ABC.
 */
class FrameFilter {
public:

    /**
     * Abstract frame filter.
     * All concrete frame filter types implement this ABC.
     * @param source_name Image SOURCE name
     * @param sink_name Image SINK name
     */
    FrameFilter(const std::string& source_name, const std::string& sink_name) :
      name("framefilt[" + source_name + "->" + sink_name + "]")
    , frame_source(source_name)
    , frame_sink(sink_name) { 

#ifdef OAT_USE_CUDA

        // Determine if a compatible device is available
        int num_devices = cv::cuda::getCudaEnabledDeviceCount();
        if (num_devices == 0) {
            throw (std::runtime_error("No GPU found or OpenCV was compiled without CUDA support."));
        }

        // Set device 
        int selected_gpu = 0; // TODO: should be user defined

        if (selected_gpu < 0 || selected_gpu > num_devices) {
            throw (std::runtime_error("Selected GPU index is invalid."));
        }

        cv::cuda::DeviceInfo gpu_info(selected_gpu);
        if (!gpu_info.isCompatible()) {
            throw (std::runtime_error("Selected GPU is not compatible with OpenCV."));
        }

        cv::cuda::setDevice(selected_gpu);

#ifndef NDEBUG
        cv::cuda::printShortCudaDeviceInfo(selected_gpu);
#endif
        
#endif // OAT_USE_CUDA

    }
    
    virtual ~FrameFilter() { }

    /**
     * Obtain raw frame from SOURCE. Apply filter function to raw frame. Publish
     * filtered frame to SINK.
     * @return SOURCE end-of-stream signal. If true, this component should exit.
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
     * Get frame filter name
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
    const std::string name;
    
    //Currently processed frame
    cv::Mat current_frame;
    
    // Frame SOURCE object for receiving raw frames
    oat::MatClient frame_source;
    
    // Frame SINK object for publishing filtered frames
    oat::MatServer frame_sink;
};

#endif	/* FRAMEFILT_H */

