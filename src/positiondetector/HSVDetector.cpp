//******************************************************************************
//* File:   HSVDetector.cpp
//* Author: Jon Newman <jpnewman snail mit dot edu>
//
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

#include "HSVDetector.h"
#include "DetectorFunc.h"

#include <limits>
#include <string>

#include <cpptoml.h>
#include <opencv2/opencv.hpp>

#include "../../lib/datatypes/Position2D.h"
#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/TOMLSanitize.h"
#include "../../lib/utility/make_unique.h"

namespace oat {

HSVDetector::HSVDetector(const std::string &frame_source_address,
                         const std::string &position_sink_address)
: PositionDetector(frame_source_address, position_sink_address)
{
    // Set required frame type
    required_color_ = PIX_HSV;
}

po::options_description HSVDetector::options() const
{
    // Update CLI options
    po::options_description local_opts;
    local_opts.add_options()
        ("h-thresh,H", po::value<std::string>(),
         "Array of ints between 0 and 256, [min,max], specifying the hue "
         "passband.")
        ("s-thresh,S", po::value<std::string>(),
         "Array of ints between 0 and 256, [min,max], specifying the "
         "saturation passband.")
        ("v-thresh,V", po::value<std::string>(),
         "Array of ints between 0 and 256, [min,max], specifying the value "
         "passband.")
        ("erode,e", po::value<int>(),
         "Contour erode kernel size in pixels (normalized box filter).")
        ("dilate,d", po::value<int>(),
         "Contour dilation kernel size in pixels (normalized box filter).")
        ("area,a", po::value<std::string>(),
         "Array of floats, [min,max], specifying the minimum and maximum "
         "object contour area in pixels^2.")
        ("tune,t",
         "If true, provide a GUI with sliders for tuning detection parameters.")
        ;

    return local_opts;
}

void HSVDetector::applyConfiguration(const po::variables_map &vm,
                                     const config::OptionTable &config_table)
{
    // Hue
    std::vector<int> h;
    if (oat::config::getArray<int, 2>(vm, config_table, "h-thresh", h)) {

        h_min_ = h[0];
        h_max_ = h[1];

        if (h_min_ < 0 || h_min_> 256 || h_max_ < 0 || h_max_ > 256)
           throw std::runtime_error("Values of h-thresh should be between 0 and 256.");
    }

    // Saturation
    std::vector<int> s;
    if (oat::config::getArray<int, 2>(vm, config_table, "s-thresh", s)) {

        s_min_ = s[0];
        s_max_ = s[1];

        if (s_min_ < 0 || s_min_> 256 || s_max_ < 0 || s_max_ > 256)
           throw std::runtime_error("Values of s-thresh should be between 0 and 256.");
    }

    // Value
    std::vector<int> v;
    if (oat::config::getArray<int, 2>(vm, config_table, "v-thresh", v)) {

        v_min_ = v[0];
        v_max_ = v[1];

        if (v_min_ < 0 || v_min_> 256 || v_max_ < 0 || v_max_ > 256)
           throw std::runtime_error("Values of v-thresh should be between 0 and 256.");
    }

    // Erode size
    oat::config::getNumericValue<int>(vm, config_table, "erode", erode_px_, 0, 50);

    // Dilate size
    oat::config::getNumericValue<int>(vm, config_table, "dilate", dilate_px_, 0, 50);

    // Min/max object area
    std::vector<double> area;
    if (oat::config::getArray<double, 2>(vm, config_table, "area", area)) {

        min_object_area_ = area[0];
        max_object_area_ = area[1];

        if (min_object_area_ >= max_object_area_)
           throw std::runtime_error("Max area should be larger than min area.");
    }

    bool tuning_on = false;
    oat::config::getValue<bool>(vm, config_table, "tune", tuning_on);

    if (tuning_on) {

        tuner_ = oat::make_unique<Tuner>(name_);

        TUNE<int>(&h_min_, "Min. hue", 0, 256, h_min_, 1);
        TUNE<int>(&h_max_, "Max. hue", 0, 256, h_max_, 1);
        TUNE<int>(&s_min_, "Min. saturation", 0, 256, s_min_, 1);
        TUNE<int>(&s_max_, "Max. saturation", 0, 256, s_max_, 1);
        TUNE<int>(&v_min_, "Min. value", 0, 256, v_min_, 1);
        TUNE<int>(&v_max_, "Max. value", 0, 256, v_max_, 1);
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
        TUNE<int>(&erode_px_, "Erode size (px)", 0, 50, erode_px_, 1);
        TUNE<int>(&dilate_px_, "Dilate size (px)", 0, 50, dilate_px_, 1);
    }
}

oat::Pose HSVDetector::detectPose(oat::Frame &frame)
{
    oat::Pose pose(frame.sample().seconds(),
                   Pose::DistanceUnit::Pixels,
                   Pose::DOF::Two,
                   Pose::DOF::Zero);

    // Threshold HSV channels (expensive operation)
    cv::Mat thresh_frame;
    cv::inRange(frame,
                cv::Scalar(h_min_, s_min_, v_min_),
                cv::Scalar(h_max_, s_max_, v_max_),
                thresh_frame);

    // Filter the resulting threshold image
    if (makeEroder(erode_px_))
        cv::erode(thresh_frame, thresh_frame, erode_element_);

    if (makeDilater(dilate_px_))
        cv::dilate(thresh_frame, thresh_frame, dilate_element_);

    // Threshold frame will be destroyed by the transform below, so we need to
    // use it to form the frame that will be shown in the tuning window here
    if (tuner_) {
        frame.setTo(0, thresh_frame == 0);
        // HACK. setTo returns a cv::Mat with no color
        frame.set_color(required_color_);
    }

    // Find the largest contour in the threshold image
    siftContours(
        thresh_frame, pose, object_area_, min_object_area_, max_object_area_);

    if (tuner_)
        tuner_->tune(frame, pose);

    return pose;
}

bool HSVDetector::makeEroder(int value)
{
    if (value <= 0)
        return false;

    erode_element_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(erode_px_, erode_px_));
    return true;
}

bool HSVDetector::makeDilater(int value)
{
    if (value <= 0)
        return false;

    dilate_element_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilate_px_, dilate_px_));
    return true;
}

} /* namespace oat */
