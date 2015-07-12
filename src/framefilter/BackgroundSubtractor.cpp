//******************************************************************************
//* File:   BackgroundSubtractor.cpp
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

#include "BackgroundSubtractor.h"

#include <string>
#include <iostream>
#include <opencv2/core.hpp>
//#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#ifdef OAT_USE_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#endif

#include "../../lib/cpptoml/cpptoml.h"
#include "../../lib/cpptoml/OatTOMLSanitize.h"
#include "../../lib/utility/IOFormat.h"

BackgroundSubtractor::BackgroundSubtractor(const std::string& source_name, const std::string& sink_name) :
  FrameFilter(source_name, sink_name) {
    
#ifdef OAT_USE_CUDA
    
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
    
#ifndef NDEBUG
    cv::cuda::printShortCudaDeviceInfo(selected_gpu);
#endif

#endif // OAT_USE_CUDA
}

void BackgroundSubtractor::configure(const std::string& config_file, const std::string& config_key) {

    // Available options
    std::vector<std::string> options {"background"};
    
    // This will throw cpptoml::parse_exception if a file 
    // with invalid TOML is provided
    cpptoml::table config;
    config = cpptoml::parse_file(config_file);

    // See if a camera configuration was provided
    if (config.contains(config_key)) {

        // Get this components configuration table
        auto this_config = config.get_table(config_key);
        
        // Check for unknown options in the table and throw if you find them
        oat::config::checkKeys(options, this_config);

        std::string background_img_path;
        if (oat::config::getValue(this_config, "background", background_img_path)) {
            background_img = cv::imread(background_img_path, CV_LOAD_IMAGE_COLOR);

            if (background_img.data == NULL) {
                throw (std::runtime_error("File \"" + background_img_path + "\" could not be read."));
            }

            background_set = true;
        }

    } else {
        throw (std::runtime_error(oat::configNoTableError(config_key, config_file)));
    }
}

void BackgroundSubtractor::setBackgroundImage(const cv::Mat& frame) {

#ifdef OAT_USE_CUDA
    background_frame.upload(frame);
    result_frame.upload(frame);
#else
    background_img = frame.clone();
#endif

    background_set = true;
}

cv::Mat BackgroundSubtractor::filter(cv::Mat& frame) {
    // Throws cv::Exception if there is a size mismatch between frames,
    // or in any case where cv assertions fail.
    
    // Only proceed with processing if we are getting a valid frame
    if (background_set) {

#ifdef OAT_USE_CUDA
        current_frame.upload(frame);
        cv::cuda::subtract(current_frame, background_frame, result_frame);
        result_frame.download(frame);
#else
        frame = frame - background_img;
#endif
    } else {

        // First image is always used as the default background image if one is
        // not provided in a configuration file
        setBackgroundImage(frame);
    }

    return frame;
}
