//******************************************************************************
//* File:   FrameMasker.cpp
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
//******************************************************************************

#include "FrameMasker.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <cpptoml.h>
#include "../../lib/utility/TOMLSanitize.h"
#include "../../lib/utility/IOFormat.h"

namespace oat {

FrameMasker::FrameMasker(const std::string &frame_source_address,
                         const std::string &frame_sink_address) :
  FrameFilter(frame_source_address, frame_sink_address)
{
    // Nothing
}

void FrameMasker::configure(const std::string &config_file, 
                            const std::string &config_key) {

    // Available options
    std::vector<std::string> options {"mask"};

    // This will throw cpptoml::parse_exception if a file
    // with invalid TOML is provided
    auto config = cpptoml::parse_file(config_file);

    // See if a configuration was provided
    if (config->contains(config_key)) {

        // Get this components configuration table
        auto this_config = config->get_table(config_key);

        // Check for unknown options in the table and throw if you find them
        oat::config::checkKeys(options, this_config);

        std::string mask_path;
        oat::config::getValue(this_config, "mask", mask_path, true);
        roi_mask_ = cv::imread(mask_path, CV_LOAD_IMAGE_GRAYSCALE);

        if (roi_mask_.data == NULL)
            throw (std::runtime_error("File \"" + mask_path + "\" could not be read."));

        mask_set_ = true;

    } else {
        throw (std::runtime_error(oat::configNoTableError(config_key, config_file)));
    }
}

void FrameMasker::filter(cv::Mat &frame) {

    // Throws cv::Exception if there is a size mismatch between mask and frames
    // received from SOURCE or in any case where setTo() assertions fail.
    if (mask_set_)
        frame.setTo(0, roi_mask_ == 0);
}

} /* namespace oat */
