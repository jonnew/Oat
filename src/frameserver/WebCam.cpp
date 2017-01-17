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

po::options_description WebCam::options() const
{
    // Update CLI options
    po::options_description local_opts;
    local_opts.add_options()
        ("index,i", po::value<int>(),
         "Camera index. Useful in multi-camera imaging "
         "configurations. Defaults to 0.")
        ("fps,r", po::value<double>(),
         "Frames to serve per second. Defaults to 20.")
        ("roi", po::value<std::string>(),
         "Four element array of unsigned ints, [x0,y0,width,height],"
         "defining a rectangular region of interest. Origin"
         "is upper left corner. ROI must fit within acquired"
         "mat size. Defaults to full sensor size.")
        ;

    return local_opts; 
}

void WebCam::applyConfiguration(const po::variables_map &vm,
                                const config::OptionTable &config_table)
{
    // Camera index
    oat::config::getNumericValue<int>(vm, config_table, "index", index_, 0);

    // Create camera and set options
    cv_camera_ = oat::make_unique<cv::VideoCapture>(index_);
    if (!cv_camera_->isOpened())
        throw std::runtime_error("Could not open webcam "
                                 + std::to_string(index_));

    // Frame rate
    double fps;
    if (oat::config::getNumericValue(vm, config_table, "fps", fps, 0.0)) {
        cv_camera_->set(cv::CAP_PROP_FPS, fps);
        if (cv_camera_->get(cv::CAP_PROP_FPS) != fps)
            std::cerr << oat::Warn("Not able to set webcam mat rate.\n");
    }

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

bool WebCam::connectToNode()
{
    cv::Mat example_frame;
    *cv_camera_ >> example_frame;

    if (use_roi_)
        example_frame = example_frame(region_of_interest_);

    frame_sink_.bind(frame_sink_address_,
                     example_frame.total() * oat::color_bytes(oat::PIX_BGR));

    shared_frame_ = frame_sink_.retrieve(
        example_frame.rows, example_frame.cols, example_frame.type(), oat::PIX_BGR);

    // Put the sample rate in the shared mat
    shared_frame_.set_rate_hz(cv_camera_->get(cv::CAP_PROP_FPS));

    return true;
}

int WebCam::process()
{
    // Frame decoding (if compression was performed) can be
    // computationally expensive. So do this outside the critical section
    cv::Mat mat;
    if (!cv_camera_->read(mat)) 
        return 1;

    if (first_frame_) 
        start_ = clock_.now();

    if (use_roi_ )
        mat = mat(region_of_interest_);

    // START CRITICAL SECTION //
    ////////////////////////////
    
    // Wait for sources to read
    frame_sink_.wait();

    // Pure SINKs increment sample count
    // NOTE: webcams have poorly controlled sample period, so it must be
    // calculated. This operation is very inexpensive
    if (first_frame_) {
        first_frame_ = false; 
    } else {
        auto time_since_start
            = std::chrono::duration_cast<Sample::Microseconds>(clock_.now()
                                                               - start_);
        shared_frame_.incrementSampleCount(time_since_start);
    }

    mat.copyTo(shared_frame_);

    // Tell sources there is new data
    frame_sink_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    return 0;
}

} /* namespace oat */
