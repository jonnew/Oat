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
                                 const std::string &position_sink_address)
: PositionDetector(frame_source_address, position_sink_address)
, tuning_image_title_(position_sink_address + "_tuning")
{
    // Set defaults for the erode and dilate blocks
    // Cannot use initializer because if these are set to 0, erode_on or
    // dilate_on must be set to false
    set_erode_size(0);
    set_dilate_size(0);

    // Set required frame type
    required_color_ = PIX_GREY;
}

void SimpleThreshold::appendOptions(po::options_description &opts)
{
    // Accepts a config file
    PositionDetector::appendOptions(opts);

    // Update CLI options
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

    opts.add(local_opts);

    // Return valid keys
    for (auto &o: local_opts.options())
        config_keys_.push_back(o->long_name());
}

void SimpleThreshold::configure(const po::variables_map &vm)
{
    // Check for config file and entry correctness
    auto config_table = oat::config::getConfigTable(vm);
    oat::config::checkKeys(config_keys_, config_table);

    // Threshold
    std::vector<int> t;
    if (oat::config::getArray<int, 2>(vm, config_table, "thresh", t)) {

        t_min_ = t[0];
        t_max_ = t[1];

        if (t_min_ < 0 || t_min_> 256 || t_max_ < 0 || t_max_ > 256)
           throw std::runtime_error("Values of thresh should be between 0 and 256.");
    }

    // Erode size
    int erode;
    if (oat::config::getNumericValue<int>(vm, config_table, "erode", erode, 0))
        set_erode_size(erode);

    // Dilate size
    int dilate;
    if (oat::config::getNumericValue<int>(vm, config_table, "dilate", dilate, 0))
        set_dilate_size(dilate);

    // Min/max object area
    std::vector<double> area;
    if (oat::config::getArray<double, 2>(vm, config_table, "area", area)) {

        min_object_area_ = area[0];
        max_object_area_ = area[1];

        // Limitation of cv::highGUI
        dummy0_ = min_object_area_;
        dummy1_ = max_object_area_;

        if (min_object_area_ >= max_object_area_)
           throw std::runtime_error("Max area should be larger than min area.");
    }

    // Tuning GUI
    oat::config::getValue<bool>(vm, config_table, "tune", tuning_on_);
}

void SimpleThreshold::detectPosition(cv::Mat &frame, oat::Position2D &position)
{
    if (tuning_on_)
        tune_frame_ = frame.clone();

    applyThreshold(frame);

    // Threshold frame will be destroyed by the transform below, so we need to use
    // it to form the frame that will be shown in the tuning window here
    if (tuning_on_)
         tune_frame_.setTo(0, threshold_frame_ == 0);

    siftContours(threshold_frame_,
                 position,
                 object_area_,
                 min_object_area_,
                 max_object_area_);

    if (tuning_on_)
        tune(tune_frame_, position);
}

void SimpleThreshold::tune(cv::Mat &frame, const oat::Position2D &position)
{
    if (!tuning_windows_created_)
        createTuningWindows();

    std::string msg = cv::format("Object not found");

    // Plot a circle representing found object
    if (position.position_valid) {

        // TODO: object_area_ is not set, so this will be 0!
        auto radius = std::sqrt(object_area_ / PI);
        cv::Point center;
        center.x = position.position.x;
        center.y = position.position.y;
        cv::circle(frame, center, radius, cv::Scalar(255), 4);
        msg = cv::format("(%d, %d) pixels",
                (int) position.position.x,
                (int) position.position.y);
    }

    int baseline = 0;
    cv::Size textSize = cv::getTextSize(msg, 1, 1, 1, &baseline);
    cv::Point text_origin(
            frame.cols - textSize.width - 10,
            frame.rows - 2 * baseline - 10);

    cv::putText(frame, msg, text_origin, 1, 1, cv::Scalar(0, 255, 0));

    cv::imshow(tuning_image_title_, frame);
    cv::waitKey(1);
}

void SimpleThreshold::applyThreshold(cv::Mat &frame)
{
    cv::inRange(frame,
                t_min_,
                t_max_,
                threshold_frame_);

    // Filter the resulting threshold image
    if (erode_on_)
        cv::erode(threshold_frame_, threshold_frame_, erode_element_);

    if (dilate_on_)
        cv::dilate(threshold_frame_, threshold_frame_, dilate_element_);
}

void SimpleThreshold::createTuningWindows()
{
#ifdef HAVE_OPENGL
    try {
        cv::namedWindow(tuning_image_title_,
                        cv::WINDOW_OPENGL & cv::WINDOW_KEEPRATIO);
    } catch (cv::Exception& ex) {
        whoWarn(name_, "OpenCV not compiled with OpenGL support. Falling back "
                       "to OpenCV's display driver.\n");
        cv::namedWindow(tuning_image_title_,
                        cv::WINDOW_NORMAL & cv::WINDOW_KEEPRATIO);
    }
#else
    cv::namedWindow(tuning_image_title_, cv::WINDOW_NORMAL);
#endif

    // Create sliders and insert them into window
    cv::createTrackbar("MIN BOUND", tuning_image_title_, &t_min_, 256);
    cv::createTrackbar("MAX BOUND", tuning_image_title_, &t_max_, 256);
    cv::createTrackbar("MIN AREA",
                       tuning_image_title_,
                       &dummy0_,
                       OAT_POSIDET_MAX_OBJ_AREA_PIX,
                       &simpleThresholdMinAreaSliderChangedCallback,
                       this);
    cv::createTrackbar("MAX AREA",
                       tuning_image_title_,
                       &dummy1_,
                       OAT_POSIDET_MAX_OBJ_AREA_PIX,
                       &simpleThresholdMaxAreaSliderChangedCallback,
                       this);
    tuning_windows_created_ = true;
    cv::createTrackbar("ERODE",
                       tuning_image_title_,
                       &erode_px_,
                       50,
                       &simpleThresholdErodeSliderChangedCallback,
                       this);
    cv::createTrackbar("DILATE",
                       tuning_image_title_,
                       &dilate_px_,
                       50,
                       &simpleThresholdDilateSliderChangedCallback,
                       this);
}

void SimpleThreshold::set_erode_size(int value)
{
    if (value > 0) {
        erode_on_ = true;
        erode_px_ = value;
        erode_element_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(erode_px_, erode_px_));
    } else {
        erode_on_ = false;
    }
}

void SimpleThreshold::set_dilate_size(int value)
{
    if (value > 0) {
        dilate_on_ = true;
        dilate_px_ = value;
        dilate_element_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilate_px_, dilate_px_));
    } else {
        dilate_on_ = false;
    }
}

// Non-member GUI callback functions
void simpleThresholdMinAreaSliderChangedCallback(int value, void *object)
{
    auto d = static_cast<SimpleThreshold *>(object);
    d->set_min_object_area(static_cast<double>(value));
}

void simpleThresholdMaxAreaSliderChangedCallback(int value, void *object)
{
    auto d = static_cast<SimpleThreshold *>(object);
    d->set_max_object_area(static_cast<double>(value));
}

void simpleThresholdErodeSliderChangedCallback(int value, void *object)
{
    auto d = static_cast<SimpleThreshold *>(object);
    d->set_erode_size(value);
}

void simpleThresholdDilateSliderChangedCallback(int value, void *object)
{
    auto d = static_cast<SimpleThreshold *>(object);
    d->set_dilate_size(value);
}

} /* namespace oat */
