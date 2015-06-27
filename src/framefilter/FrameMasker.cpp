//******************************************************************************
//* File:   FrameMasker.cpp
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

#include "FrameMasker.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "../../lib/cpptoml/cpptoml.h"
#include "../../lib/utility/IOFormat.h"

FrameMasker::FrameMasker(const std::string& source_name, const std::string& sink_name, bool invert_mask) :
  FrameFilter(source_name, sink_name)
, invert_mask(false) { }

void FrameMasker::configure(const std::string& config_file, const std::string& config_key) {

    // This will throw cpptoml::parse_exception if a file 
    // with invalid TOML is provided
    cpptoml::table config;
    config = cpptoml::parse_file(config_file);

    // See if a configuration was provided
    if (config.contains(config_key)) {

        auto this_config = *config.get_table(config_key);

        std::string mask_path;
        if (this_config.contains("mask")) {
            
            if (!this_config.get("mask")->is_value()) {
                throw (std::runtime_error(oat::configValueError(
                       "mask", config_key, config_file, "must be a TOML string "
                        "specifying a path to a mask image."))
                      );
            }
            
            mask_path = *this_config.get_as<std::string>("mask");
            roi_mask = cv::imread(mask_path, CV_LOAD_IMAGE_GRAYSCALE);

            if (roi_mask.data == NULL) {
                throw (std::runtime_error("File \"" + mask_path + "\" could not be read."));
            }

            mask_set = true;
        }
    } else {
        throw (std::runtime_error(oat::configNoTableError(config_key, config_file)));
    }
}

cv::Mat FrameMasker::filter(cv::Mat& frame) {

    // Throws cv::Exception if there is a size mismatch between mask and frames
    // received from SOURCE with custom message, or in any case where setTo()
    // assertions fail.
    if (mask_set) {

        if (roi_mask.size != frame.size) {
            std::string error_message = "Mask frame and frames from SOURCE do not have equal sizes";
            CV_Error(cv::Error::StsBadSize, error_message);
        }

        frame.setTo(0, roi_mask == 0);
    }

    return frame;
}




