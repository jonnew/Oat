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
#include "Tuner.h"

#include <string>

#include <cpptoml.h>
#include <opencv2/cvconfig.h>
#include <opencv2/opencv.hpp>

#include "../../lib/datatypes/Position2D.h"
#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/TOMLSanitize.h"
#include "../../lib/utility/make_unique.h"

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
    bool tuning_on = false;
    oat::config::getValue<bool>(vm, config_table, "tune", tuning_on);

    if (tuning_on) {

        tuner_ = oat::make_unique<Tuner>(name_);

        TUNE<int>(&difference_intensity_threshold_,
                  "Diff. thresh. (px)",
                  0,
                  256,
                  difference_intensity_threshold_,
                  1);
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

oat::Pose DifferenceDetector::detectPose(oat::Frame &frame)
{
    oat::Pose pose(frame.sample().seconds(),
                   Pose::DistanceUnit::Pixels,
                   Pose::DOF::Two,
                   Pose::DOF::Zero);

    cv::Mat thresh_frame;
    if (last_frame_set_) {
        cv::absdiff(frame, last_frame_, thresh_frame);
        cv::threshold(thresh_frame,
                      thresh_frame,
                      difference_intensity_threshold_,
                      255,
                      cv::THRESH_BINARY);

        if (makeBlur(blur_px_))
            cv::blur(thresh_frame, thresh_frame, blur_size_);

        last_frame_ = frame.clone(); // Get a copy of the last image
    } else {
        thresh_frame = frame.clone();
        last_frame_ = frame.clone();
        last_frame_set_ = true;
    }

    // Threshold frame will be destroyed by the transform below, so we need to
    // use it to form the frame that will be shown in the tuning window here
    if (tuner_) {
        frame.setTo(0, thresh_frame == 0);
        // HACK. setTo returns a cv::Mat with no color
        frame.set_color(required_color_);
    }

    // NB: Mutates thresh_frame
    siftContours(
        thresh_frame, pose, object_area_, min_object_area_, max_object_area_);

    if (tuner_)
        tuner_->tune(frame, pose);

    return pose;
}

bool DifferenceDetector::makeBlur(int value)
{
    if (value <= 0)
        return false;

    blur_size_ = cv::Size(value, value);
    return true;
}

} /* namespace oat */
