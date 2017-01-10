//******************************************************************************
//* File:   UCLAMiniscope.cpp
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

#include "UCLAMiniscope.h"

#include <chrono>
#include <string>
#include <opencv2/core/mat.hpp>

#include "../../lib/utility/TOMLSanitize.h"
#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/make_unique.h"

namespace oat {

po::options_description UCLAMiniscope::options() const
{
    // Update CLI options
    po::options_description local_opts;
    local_opts.add_options()
        ("index,i", po::value<int>(),
         "Camera index. Useful in multi-camera imaging "
         "configurations. Defaults to 0.")
        ("fps,r", po::value<double>(),
         "Frames to serve per second. Defaults to 20.")
        ("led-intensity,l", po::value<double>(),
         "Float, 0 to 1.0, indicating relative LED intensity. Defaults to "
         "0.1.")
        ("exposure,e", po::value<double>(),
         "Float, 0 to 1.0, indicating relative shutter open time. Defaults to "
         "0.5.")
        ("gain,g", po::value<double>(),
         "Float, 0 to 1.0, indicating relative sensor gain. Defaults to 0.5.")
        ("roi", po::value<std::string>(),
         "Four element array of unsigned ints, [x0,y0,width,height],"
         "defining a rectangular region of interest. Origin"
         "is upper left corner. ROI must fit within acquired"
         "frame size. Defaults to full sensor size.")
        ;

    return local_opts;
}

void UCLAMiniscope::applyConfiguration(const po::variables_map &vm,
                                const config::OptionTable &config_table)
{
    // Camera index
    oat::config::getNumericValue<int>(vm, config_table, "index", index_, 0);

    // Create camera and set options
    camera_ = oat::make_unique<cv::VideoCapture>(index_);
    if (!camera_->isOpened())
        throw std::runtime_error("Could not open miniscope "
                                 + std::to_string(index_));
    // LED intensity
    double intensity = 0.1;
    oat::config::getNumericValue<double>(
        vm, config_table, "led-intensity", intensity, 0, 1.0);
    std::cout << intensity << std::endl;
    setLEDIntensity(0, intensity);

    // Exposure
    double exp = 0.5;
    oat::config::getNumericValue<double>(
        vm, config_table, "exposure", exp, 0, 1.0);
    camera_->set(CV_CAP_PROP_BRIGHTNESS, 100 * exp);

    // Gain
    double gain = 0.5;
    oat::config::getNumericValue<double>(
        vm, config_table, "exposure", exp, 0, 1.0);
    auto gain_i = static_cast<int>(gain * 64); // Max value for MT9V032
    if (gain_i >= 32 && (gain_i % 2) == 1)
        gain_i++; // (from miniscope software: Gains between 32 and 64 must
                // be even for MT9V032, which is the imagin chip in the scope)
    camera_->set(CV_CAP_PROP_GAIN, gain_i);

    // Frame rate
    double fps;
    if (oat::config::getNumericValue(vm, config_table, "fps", fps, 0.0)) {
        camera_->set(cv::CAP_PROP_FPS, fps);
        if (camera_->get(cv::CAP_PROP_FPS) != fps)
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

bool UCLAMiniscope::connectToNode()
{
    cv::Mat example_frame;
    *camera_ >> example_frame;

    if (use_roi_)
        example_frame = example_frame(region_of_interest_);

    frame_sink_.bind(frame_sink_address_,
                     example_frame.total() * oat::color_bytes(oat::PIX_BGR));

    shared_frame_ = frame_sink_.retrieve(
        example_frame.rows, example_frame.cols, example_frame.type(), oat::PIX_BGR);

    // Put the sample rate in the shared mat
    shared_frame_.set_rate_hz(camera_->get(cv::CAP_PROP_FPS));

    return true;
}

int UCLAMiniscope::process()
{
    // Frame decoding (if compression was performed) can be
    // computationally expensive. So do this outside the critical section
    cv::Mat mat;
    if (!camera_->read(mat))
        return 1;

    if (use_roi_ )
        mat = mat(region_of_interest_);

    if (first_frame_)
        start_ = clock_.now();

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

void UCLAMiniscope::setLEDIntensity(size_t index, double value)
{
    // Only one LED, currently
    (void)index;

    // Full range is 4095
    // MSB-2 and MSB-3 need to be 1 for some reason
    //auto reg = static_cast<uint8_t>(value * 255);
    //
    int lala = (int)(value * 100);
    uint16_t toto = ((uint16_t)(lala * (0x0FFF) / 100)) | (0x3000); // Full range
    std::cout << std::hex << toto << std::endl;
    camera_->set(CV_CAP_PROP_HUE, (toto >> 4) & 0x00FF);

    //camera_->set(CV_CAP_PROP_HUE, reg);
}

} /* namespace oat */
