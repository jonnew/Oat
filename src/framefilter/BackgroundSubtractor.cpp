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
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "../../lib/cpptoml/cpptoml.h"

BackgroundSubtractor::BackgroundSubtractor(const std::string& source_name, const std::string& sink_name) :
  FrameFilter(source_name, sink_name) {
}

void BackgroundSubtractor::configure(const std::string& config_file, const std::string& config_key) {

    // This will throw cpptoml::parse_exception if a file 
    // with invalid TOML is provided
    cpptoml::table config;
    config = cpptoml::parse_file(config_file);

    // See if a camera configuration was provided
    if (config.contains(config_key)) {

        auto this_config = *config.get_table(config_key);

        std::string background_img_path;
        if (this_config.contains("background")) {
            background_img_path = *this_config.get_as<std::string>("background");

            background_img = cv::imread(background_img_path, CV_LOAD_IMAGE_COLOR);

            if (background_img.data == NULL) {
                throw (std::runtime_error("File \"" + background_img_path + "\" could not be read."));
            }

            background_set = true;
        }
    } else {
        throw ( std::runtime_error(
                "No background subtractor configuration named " + config_key +
                " was provided in the configuration file " + config_file)
                );
    }
} 

/**
 * Set the background image to be used during subsequent subtraction operations.
 * The frame_source must have previously populated the the shared cv::Mat object.
 * 
 */
void BackgroundSubtractor::setBackgroundImage(const cv::Mat& frame) {

    background_img = frame.clone();
    background_set = true;
}

/**
 * Subtract a previously set background image from an input image to produce
 * the output matrix.
 * 
 */
cv::Mat BackgroundSubtractor::filter(cv::Mat& frame) {
    // Throws cv::Exception if there is a size mismatch between frames,
    // or in any case where cv assertions fail.
    
    // Only proceed with processing if we are getting a valid frame
    if (background_set) {

        if (background_img.size != frame.size) {
            std::string error_message = "Background frame and frames from SOURCE do not have equal sizes";
            CV_Error(cv::Error::StsBadSize, error_message);

        }

        frame = frame - background_img;

    } else {

        // First image is always used as the default background image
	// if one is not provided in a configuration file
        setBackgroundImage(frame);
    }

    return frame;
}
