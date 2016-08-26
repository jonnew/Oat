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

#include <cpptoml.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>

#include "../../lib/utility/TOMLSanitize.h"
#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/make_unique.h"

namespace oat {

WebCam::WebCam(const std::string &frame_sink_name) :
  FrameServer(frame_sink_name)
{
    config_keys_ = {"index",
                    "fps",
                    "roi"};
}

void WebCam::appendOptions(po::options_description &opts) const {

    // Accepts default options
    FrameServer::appendOptions(opts);

    // Update CLI options
    opts.add_options()
        ("index,i", po::value<int64_t>(),
         "Camera index. Defaults to 0. Useful in multi-camera imaging configurations.")
        //("fps,r", po::value<double>(),
        // "Frames to serve per second.")
        ("roi {CF}", po::value<std::string>(),
         "Four element array of ints, [x0 y0 width height],"
         "defining a rectangular region of interest. Origin"
         "is upper left corner. ROI must fit within acquired"
         "frame size.")
        ;
}

void WebCam::configure(const po::variables_map &vm) {

    // Check for config file and entry correctness
    auto config_table = oat::config::getConfigTable(vm);
    oat::config::checkKeys(config_keys_, config_table);

    // Camera index
    oat::config::getNumericValue<size_t>(
        vm, config_table, "index", index_, 0
    );

    //// Frame rate
    //if (oat::config::getNumericValue(vm, config_table, "fps", frames_per_second_, 0.0))
    //    calculateFramePeriod();

    // ROI
    oat::config::Array roi;
    if (oat::config::getArray(config_table, "roi", roi, 4, false)) {

        use_roi_ = true;
        auto roi_arr = roi->array_of<int64_t>();

        region_of_interest_.x      = static_cast<int>(roi_arr[0]->get());
        region_of_interest_.y      = static_cast<int>(roi_arr[1]->get());
        region_of_interest_.width  = static_cast<int>(roi_arr[2]->get());
        region_of_interest_.height = static_cast<int>(roi_arr[3]->get());
    } 
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

bool WebCam::process() {

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

    // Increment sample count
    if (shared_frame_.sample().count() == 0)
        start_ = clock_.now();

    auto period =
        std::chrono::duration_cast<Sample::Microseconds>(clock_.now() - start_);
    shared_frame_.sample().incrementCount(period);

    // Tell sources there is new data
    frame_sink_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    return frame_empty_;
}

} /* namespace oat */
