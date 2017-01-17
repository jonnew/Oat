//******************************************************************************
//* File:   SimpleThreshold.cpp
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

#include "SimpleThreshold.h"
#include "DetectorFunc.h"

#include <string>
#include <opencv2/cvconfig.h>
#include <opencv2/opencv.hpp>
#include <cpptoml.h>

#include "../../lib/datatypes/Position2D.h"
#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/TOMLSanitize.h"

namespace oat {

SimpleThreshold::SimpleThreshold(const std::string &frame_source_address,
                                 const std::string &pose_sink_address)
: PositionDetector(frame_source_address, pose_sink_address)
{
    // Set required frame type
    required_color_ = PIX_GREY;
}

po::options_description SimpleThreshold::options() const
{
    po::options_description local_opts;
    local_opts.add_options()
        ("thresh,T", po::value<std::string>(),
         "Array of ints between 0 and 256, [min,max], specifying the "
         "intensity passband.")
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

void SimpleThreshold::applyConfiguration(
    const po::variables_map &vm, const config::OptionTable &config_table)
{
    // Threshold
    std::vector<int> t;
    if (oat::config::getArray<int, 2>(vm, config_table, "thresh", t)) {

        t_min_ = t[0];
        t_max_ = t[1];

        if (t_min_ < 0 || t_min_> 256 || t_max_ < 0 || t_max_ > 256)
           throw std::runtime_error("Values of thresh should be between 0 and 256.");
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

    // Tuning GUI
    oat::config::getValue<bool>(vm, config_table, "tune", tuning_on_);

    if (tuning_on_) {
        TUNE<int>(&t_min_, "Min. value", 0, 256, t_min_, 1);
        TUNE<int>(&t_max_, "Max. value", 0, 256, t_max_, 1);
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

void SimpleThreshold::detectPosition(oat::Frame &frame, oat::Pose &pose)
{
    // Must do this here because inRange slices the frame.
    if (tuning_on_)
        tuning_frame_ = frame.clone();

    cv::inRange(frame,
                t_min_,
                t_max_,
                threshold_frame_);

    // Filter the resulting threshold image
    if (makeEroder(erode_px_))
        cv::erode(threshold_frame_, threshold_frame_, erode_element_);

    if (makeDilater(dilate_px_))
        cv::dilate(threshold_frame_, threshold_frame_, dilate_element_);

    // Threshold frame will be destroyed by the transform below, so we need to use
    // it to form the frame that will be shown in the tuning window here
    if (tuning_on_)
        tuning_frame_.setTo(0, threshold_frame_ == 0);

    siftContours(threshold_frame_,
                 pose,
                 object_area_,
                 min_object_area_,
                 max_object_area_);
}

bool SimpleThreshold::makeEroder(int value)
{
    if (value <= 0)
        return false;

    erode_element_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(erode_px_, erode_px_));
    return true;
}

bool SimpleThreshold::makeDilater(int value)
{
    if (value <= 0)
        return false;

    dilate_element_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilate_px_, dilate_px_));
    return true;
}

} /* namespace oat */
