//******************************************************************************
//* File:   WebCam.cpp
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

#include "WebCam.h"

#include <string>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>

#include <cpptoml.h>

#include "../../lib/utility/TOMLSanitize.h"
#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/make_unique.h"

namespace oat {

WebCam::WebCam(const std::string &frame_sink_name) :
  FrameServer(frame_sink_name)
, index_(0)
{
    // Nothing
}

void WebCam::connectToNode() {

    cv_camera_ = std::make_unique<cv::VideoCapture>(index_);

    cv::Mat example_frame;
    *cv_camera_ >> example_frame;

    if (use_roi_)
        example_frame = example_frame(region_of_interest_);

    frame_sink_.bind(frame_sink_address_,
            example_frame.total() * example_frame.elemSize());

    shared_frame_ = frame_sink_.retrieve(
            example_frame.rows, example_frame.cols, example_frame.type());

}

bool WebCam::serveFrame() {

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sources to read
    frame_sink_.wait();

    if (!use_roi_) {
            
        *cv_camera_ >> shared_frame_;
        frame_empty_ = shared_frame_.empty();

    } else {

        oat::Frame to_crop;
        *cv_camera_ >> to_crop;
        if (!(frame_empty_ = to_crop.empty()))
            to_crop = to_crop(region_of_interest_);
        to_crop.copyTo(shared_frame_);
    }

    // Update sample count
    shared_frame_.sample() = internal_sample_;

    // Tell sources there is new data
    frame_sink_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // Pure SINKs increment sample count
    // NOTE: webcams have unctrolled sample period, so it must be calculated.
    if (internal_sample_.count() == 0)
        start_ = clock_.now();

    auto runtime = 
        std::chrono::duration_cast<Sample::Microseconds>(clock_.now() - start_);
    internal_sample_.incrementCount(runtime);

    return frame_empty_;
}

void WebCam::configure() { }

void WebCam::configure(const std::string& config_file, const std::string& config_key) {

    // Available options
    std::vector<std::string> options {"index", "roi"};

    // This will throw cpptoml::parse_exception if a file
    // with invalid TOML is provided
    auto config = cpptoml::parse_file(config_file);

    // See if a camera configuration was provided
    if (config->contains(config_key)) {

        // Get this components configuration table
        auto this_config = config->get_table(config_key);

        // Check for unknown options in the table and throw if you find them
        oat::config::checkKeys(options, this_config);

        // Set the camera index
        oat::config::getValue(this_config, "index", index_, MIN_INDEX);
        //cv_camera_ = std::make_unique<cv::VideoCapture>(index_);

        // Set the ROI
        oat::config::Table roi;
        if (oat::config::getTable(this_config, "roi", roi)) {

            int64_t val;
            oat::config::getValue(roi, "x_offset", val, (int64_t)0, true);
            region_of_interest_.x = val;
            oat::config::getValue(roi, "y_offset", val, (int64_t)0, true);
            region_of_interest_.y = val;
            oat::config::getValue(roi, "width", val, (int64_t)0, true);
            region_of_interest_.width = val;
            oat::config::getValue(roi, "height", val, (int64_t)0, true);
            region_of_interest_.height = val;
            use_roi_ = true;

        }

    } else {
        throw (std::runtime_error(oat::configNoTableError(config_key, config_file)));
    }
}

} /* namespace oat */
