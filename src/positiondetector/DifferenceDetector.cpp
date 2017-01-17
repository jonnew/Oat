//******************************************************************************
//* File:   DifferenceDetector.cpp
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
//****************************************************************************

#include "DifferenceDetector.h"
#include "DetectorFunc.h"

#include <string>
#include <opencv2/cvconfig.h>
#include <opencv2/opencv.hpp>
#include <cpptoml.h>

#include "../../lib/datatypes/Position2D.h"
#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/TOMLSanitize.h"

namespace oat {

DifferenceDetector::DifferenceDetector(const std::string &frame_source_address,
                                       const std::string &position_sink_address)
: PositionDetector(frame_source_address, position_sink_address)
{
    // Set required frame type
    required_color_ = PIX_GREY;
}

po::options_description DifferenceDetector::options() const
{
    po::options_description local_opts;
    local_opts.add_options()
        ("diff-threshold,d", po::value<int>(),
         "Intensity difference threshold to consider an object contour.")
        ("blur,b", po::value<int>(),
         "Blurring kernel size in pixels (normalized box filter).")
        ("area,a", po::value<std::string>(),
         "Array of floats, [min,max], specifying the minimum and maximum "
         "object contour area in pixels^2.")
        ("tune,t",
         "If true, provide a GUI with sliders for tuning detection "
         "parameters.")
        ;

    return local_opts;
}

void DifferenceDetector::applyConfiguration(
    const po::variables_map &vm, const config::OptionTable &config_table)
{
    // Difference threshold
    oat::config::getNumericValue<int>(
        vm, config_table, "diff-threshold", difference_intensity_threshold_, 0
    );

    // Blur
    oat::config::getNumericValue<int>(vm, config_table, "blur", blur_px_, 0, 50);

    // Min/max object area
    std::vector<double> area;
    if (oat::config::getArray<double, 2>(vm, config_table, "area", area)) {

        min_object_area_ = area[0];
        max_object_area_ = area[1];

        if (min_object_area_ >= max_object_area_)
           throw std::runtime_error("Max area should be larger than min area.");
    }

    // Tuning GUI
    oat::config::getValue<bool>(vm, config_table, "tune", tuning_on_);

    if (tuning_on_) {
        TUNE<double>(&min_object_area_,
                     "Min. area (px^2)",
                     0,
                     OAT_POSIDET_MAX_OBJ_AREA_PIX,
                     min_object_area_,
                     1);
        TUNE<double>(&max_object_area_,
                     "Max. area (px^2)",
                     0,
                     OAT_POSIDET_MAX_OBJ_AREA_PIX,
                     max_object_area_,
                     1);
        TUNE<int>(&blur_px_, "Blur size (px)", 0, 50, blur_px_, 1);
    }
}

void DifferenceDetector::detectPosition(oat::Frame &frame, oat::Pose &position)
{
    if (tuning_on_)
        tuning_frame_ = frame.clone();

    applyThreshold(frame);

    // Threshold frame will be destroyed by the transform below, so we need to use
    // it to form the frame that will be shown in the tuning window here
    if (tuning_on_)
        tuning_frame_ = tuning_frame_.setTo(0, threshold_frame_ == 0);

    siftContours(threshold_frame_,
                 position,
                 object_area_,
                 min_object_area_,
                 max_object_area_);

}

void DifferenceDetector::applyThreshold(cv::Mat &frame) {

    if (last_image_set_) {
        cv::absdiff(frame, last_image_, threshold_frame_);
        cv::threshold(threshold_frame_,
                      threshold_frame_,
                      difference_intensity_threshold_,
                      255,
                      cv::THRESH_BINARY);

        if (makeBlur(blur_px_))
            cv::blur(threshold_frame_, threshold_frame_, blur_size_);


        last_image_ = frame.clone(); // Get a copy of the last image
    } else {
        threshold_frame_ = frame.clone();
        last_image_ = frame.clone();
        last_image_set_ = true;
    }
}

bool DifferenceDetector::makeBlur(int value)
{
    if (value <= 0)
        return false;

    blur_size_ = cv::Size(value, value);
    return true;
}

} /* namespace oat */
