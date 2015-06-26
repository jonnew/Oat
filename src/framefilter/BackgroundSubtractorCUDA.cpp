//******************************************************************************
//* File:   BackgroundSubtractorCUDA.cpp
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

#include "BackgroundSubtractorCUDA.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>

BackgroundSubtractorCUDA::BackgroundSubtractorCUDA(const std::string& source_name, const std::string& sink_name) :
  FrameFilter(source_name, sink_name) {
    
    // Determine if a compatible device is available
    int num_devices = cv::cuda::getCudaEnabledDeviceCount();
    if (num_devices == 0)
    {
        throw(std::runtime_error("No GPU found or OpenCV was compiled without CUDA support."));
    }
    
    // Set device 
    int selected_gpu = 0; // TODO: should be user defined
    
    if (selected_gpu < 0 || selected_gpu > num_devices)
    {
        throw(std::runtime_error("Selected GPU index is invalid."));
    } 
    
    cv::cuda::DeviceInfo gpu_info(selected_gpu);
    if (!gpu_info.isCompatible())
    {
        throw(std::runtime_error("Selected GPU is not compatible with OpenCV."));
    } 
    
    cv::cuda::setDevice(selected_gpu);
    
#ifndef DEBUG
    cv::cuda::printShortCudaDeviceInfo(selected_gpu);
#endif
}

void BackgroundSubtractorCUDA::configure(const std::string& config_file, const std::string& config_key) {
    
}

void BackgroundSubtractorCUDA::setBackgroundImage(const cv::Mat& frame) {

    background_frame.upload(frame);
    result_frame.upload(frame);
    background_set = true;
    
}

cv::Mat BackgroundSubtractorCUDA::filter(cv::Mat& frame) {
    // Throws cv::Exception if there is a size mismatch between frames,
    // or in any case where cv assertions fail.
    
    // Only proceed with processing if we are getting a valid frame
    if (background_set) {

        current_frame.upload(frame);
        cv::cuda::subtract(current_frame, background_frame, result_frame);
        result_frame.download(frame);

    } else {

        // First image is always used as the default background image
	// if one is not provided in a configuration file
        setBackgroundImage(frame);
    }

    return frame;
}