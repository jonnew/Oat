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

#include <chrono>
#include <string>
#include <opencv2/core/mat.hpp>

#include "../../lib/utility/TOMLSanitize.h"
#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/make_unique.h"

namespace oat {

WebCam::WebCam(const std::string &sink_name) :
  FrameServer(sink_name)
{
    // Nothing
}

void WebCam::appendOptions(po::options_description &opts) {

    // Accepts default options
    FrameServer::appendOptions(opts);

    // Update CLI options
    po::options_description local_opts;
    local_opts.add_options()
        ("index,i", po::value<size_t>(),
         "Camera index. Defaults to 0. Useful in multi-camera imaging configurations.")
        ("roi", po::value<std::string>(),
         "Four element array of unsigned ints, [x0,y0,width,height],"
         "defining a rectangular region of interest. Origin"
         "is upper left corner. ROI must fit within acquired"
         "frame size. Defaults to full sensor size.")
        ;

    opts.add(local_opts);

    // Return valid keys
    for (auto &o: local_opts.options())
        config_keys_.push_back(o->long_name());
}

void WebCam::configure(const po::variables_map &vm) {

    // Check for config file and entry correctness
    auto config_table = oat::config::getConfigTable(vm);
    oat::config::checkKeys(config_keys_, config_table);

    // Camera index
    oat::config::getNumericValue<size_t>(
        vm, config_table, "index", index_, 0
    );

    // ROI
    std::vector<size_t> roi;
    if (oat::config::getArray<size_t, 4>(vm, config_table, "roi", roi)) {

        use_roi_ = true;
        region_of_interest_.x      = roi[0];
        region_of_interest_.y      = roi[1];
        region_of_interest_.width  = roi[2];
        region_of_interest_.height = roi[3];
    }
}

void WebCam::connectToNode() {

    cv_camera_ = std::make_unique<cv::VideoCapture>(index_);
    if (!cv_camera_->isOpened())
        throw (std::runtime_error("Could not open webcam " + std::to_string(index_)));

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

    bool frame_empty = false;

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sources to read
    frame_sink_.wait();

    if (!use_roi_) {

        *cv_camera_ >> shared_frame_;
        frame_empty = shared_frame_.empty();

    } else {

        oat::Frame to_crop;
        *cv_camera_ >> to_crop;
        frame_empty = to_crop.empty();
        if (!frame_empty) {
            to_crop = to_crop(region_of_interest_);
            to_crop.copyTo(shared_frame_);
        }
    }

    //// Update sample count
    shared_frame_.sample() = internal_sample_;

    // Tell sources there is new data
    frame_sink_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // Pure SINKs increment sample count
    // NOTE: webcams have unctrolled sample period, so it must be calculated.
    if (internal_sample_.count() == 0)
        start_ = clock_.now();

    auto period =
        std::chrono::duration_cast<Sample::Microseconds>(clock_.now() - start_);
    internal_sample_.incrementCount(period);

    return frame_empty;
}

} /* namespace oat */
