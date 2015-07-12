//******************************************************************************
//* File:   WebCam.cpp 
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

#include "WebCam.h"

#include <string>

#include "../../lib/cpptoml/cpptoml.h"
#include "../../lib/cpptoml/OatTOMLSanitize.h"
#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/make_unique.h"

WebCam::WebCam(std::string frame_sink_name) :
  FrameServer(frame_sink_name)
, index(0)
, cv_camera(std::make_unique<cv::VideoCapture>(index)){ }

void WebCam::grabFrame(cv::Mat& frame) {
    *cv_camera >> frame;
}

void WebCam::configure() { }
void WebCam::configure(const std::string& config_file, const std::string& config_key) {

    // Available options
    std::vector<std::string> options {"index"};
    
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
        
        // Set the camera index
        oat::config::getValue(this_config, "index", index, min_index);
        cv_camera = std::make_unique<cv::VideoCapture>(index);
        
    } else {
        throw (std::runtime_error(oat::configNoTableError(config_key, config_file)));
    }
}
