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

#include <string>
#include <limits>
#include <opencv2/opencv.hpp>
#include <cpptoml.h>

#include "../../lib/datatypes/Position2D.h"
#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/TOMLSanitize.h"

namespace oat {

HSVDetector::HSVDetector(const std::string &frame_source_address,
                         const std::string &position_sink_address)
: PositionDetector(frame_source_address, position_sink_address)
, tuning_image_title_(position_sink_address + "_tuning")
{
    // Set defaults for the erode and dilate blocks
    // Cannot use initializer because if these are set to 0, erode_on or
    // dilate_on must be set to false
    set_erode_size(0);
    set_dilate_size(10);

    // Set required frame type
    explicit_type_ = CV_8UC3;
}

void HSVDetector::appendOptions(po::options_description &opts)
{
    // Accepts a config file
    PositionDetector::appendOptions(opts);

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

    opts.add(local_opts);

    // Return valid keys
    for (auto &o : local_opts.options())
        config_keys_.push_back(o->long_name());
}

void HSVDetector::configure(const po::variables_map &vm)
{
    // Check for config file and entry correctness
    auto config_table = oat::config::getConfigTable(vm);
    oat::config::checkKeys(config_keys_, config_table);

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

void HSVDetector::detectPosition(cv::Mat &frame, oat::Position2D &position)
{
    // Transform frame to HSV
    // (Extremely expensive operation)
    cv::cvtColor(frame, frame, cv::COLOR_BGR2HSV);

    // Threshold HSV channels
    // (Very expensive operation)
    cv::inRange(frame,
                cv::Scalar(h_min_, s_min_, v_min_),
                cv::Scalar(h_max_, s_max_, v_max_),
                threshold_frame_);

    // Filter the resulting threshold image
    if (erode_on_)
        cv::erode(threshold_frame_, threshold_frame_, erode_element_);

    if (dilate_on_)
        cv::dilate(threshold_frame_, threshold_frame_, dilate_element_);

    // Threshold frame will be destroyed by the transform below, so we need to use
    // it to form the frame that will be shown in the tuning window here
    if (tuning_on_)
        frame.setTo(0, threshold_frame_ == 0).clone();

    // Find the largest contour in the threshold image
    siftContours(threshold_frame_,
                 position,
                 object_area_,
                 min_object_area_,
                 max_object_area_);

    // Use the GUI tuner if requested
    if (tuning_on_)
        tune(frame, position);
}

void HSVDetector::tune(cv::Mat &frame, const oat::Position2D &position)
{
    if (!tuning_windows_created_)
        createTuningWindows();

    std::string msg = cv::format("Object not found");

    // Plot a circle representing found object
    if (position.position_valid) {
        auto radius = std::sqrt(object_area_ / PI);
        cv::Point center;
        center.x = position.position.x;
        center.y = position.position.y;
        cv::circle(frame, center, radius, cv::Scalar(0, 0, 255), 4);
        msg = cv::format("(%d, %d) pixels",
                         (int)position.position.x,
                         (int)position.position.y);
    }

    int baseline = 0;
    cv::Size textSize = cv::getTextSize(msg, 1, 1, 1, &baseline);
    cv::Point text_origin(frame.cols - textSize.width - 10,
                          frame.rows - 2 * baseline - 10);

    cv::putText(frame, msg, text_origin, 1, 1, cv::Scalar(0, 255, 0));

    cv::imshow(tuning_image_title_, frame);
    cv::waitKey(1);
}

void HSVDetector::createTuningWindows()
{
#ifdef HAVE_OPENGL
    try {
        cv::namedWindow(tuning_image_title_, cv::WINDOW_OPENGL & cv::WINDOW_KEEPRATIO);
    } catch (cv::Exception& ex) {
        whoWarn(name_, "OpenCV not compiled with OpenGL support. Falling back to OpenCV's display driver.\n");
        cv::namedWindow(name_, cv::WINDOW_NORMAL & cv::WINDOW_KEEPRATIO);
    }
#else
    cv::namedWindow(tuning_image_title_, cv::WINDOW_NORMAL);
#endif

    // Create sliders and insert them into window
    cv::createTrackbar("H MIN", tuning_image_title_, &h_min_, 256);
    cv::createTrackbar("H MAX", tuning_image_title_, &h_max_, 256);
    cv::createTrackbar("S MIN", tuning_image_title_, &s_min_, 256);
    cv::createTrackbar("S MAX", tuning_image_title_, &s_max_, 256);
    cv::createTrackbar("V MIN", tuning_image_title_, &v_min_, 256);
    cv::createTrackbar("V MAX", tuning_image_title_, &v_max_, 256);
    cv::createTrackbar("MIN AREA",
                       tuning_image_title_,
                       &dummy0_,
                       OAT_POSIDET_MAX_OBJ_AREA_PIX,
                       &hsvDetectorMinAreaSliderChangedCallback,
                       this);
    cv::createTrackbar("MAX AREA",
                       tuning_image_title_,
                       &dummy1_,
                       OAT_POSIDET_MAX_OBJ_AREA_PIX,
                       &hsvDetectorMaxAreaSliderChangedCallback,
                       this);
    cv::createTrackbar("ERODE",
                       tuning_image_title_,
                       &erode_px_,
                       50,
                       &hsvDetectorErodeSliderChangedCallback,
                       this);
    cv::createTrackbar("DILATE",
                       tuning_image_title_,
                       &dilate_px_,
                       50,
                       &hsvDetectorDilateSliderChangedCallback,
                       this);

    tuning_windows_created_ = true;
}

void HSVDetector::set_erode_size(int value)
{
    if (value > 0) {
        erode_on_ = true;
        erode_px_ = value;
        erode_element_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(erode_px_, erode_px_));
    } else {
        erode_on_ = false;
    }
}

void HSVDetector::set_dilate_size(int value)
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
void hsvDetectorMinAreaSliderChangedCallback(int value, void *object)
{
    auto hsv_detector = static_cast<HSVDetector *>(object);
    hsv_detector->set_min_object_area(static_cast<double>(value));
}

void hsvDetectorMaxAreaSliderChangedCallback(int value, void *object)
{
    auto hsv_detector = static_cast<HSVDetector *>(object);
    hsv_detector->set_max_object_area(static_cast<double>(value));
}

void hsvDetectorErodeSliderChangedCallback(int value, void *object)
{
    auto hsv_detector = static_cast<HSVDetector *>(object);
    hsv_detector->set_erode_size(value);
}

void hsvDetectorDilateSliderChangedCallback(int value, void *object)
{
    auto hsv_detector = static_cast<HSVDetector *>(object);
    hsv_detector->set_dilate_size(value);
}

} /* namespace oat */

// NOTE: This code was from a leftover functional CUDA implementation that did not
// give me the performance gains I was hoping for. I'm leaving it here in case
// I figure something out that changes my mind about it.
//
//#ifdef NOIMP_OAT_USE_CUDA
//#include <opencv2/core/cuda.hpp>
//#include <opencv2/cudaarithm.hpp>
//#include <opencv2/cudafilters.hpp>
//#include <opencv2/cudaimgproc.hpp>
//#endif
//
//#ifdef NOIMP_OAT_USE_CUDA
//    createHSVLUT();
//#endif
//#ifdef NOIMP_OAT_USE_CUDA
//    cv::cuda::GpuMat hsv_image_, lut_frame, threshold_frame_;
//    cv::Mat search_frame;
//    cv::Mat tuning_image;
//    std::vector<cv::cuda::GpuMat> hsv_channels;
//    cv::Ptr<cv::cuda::LookUpTable> hsv_lut;
//    cv::Ptr<cv::cuda::Filter> erode_filter;
//    cv::Ptr<cv::cuda::Filter> dilate_filter;
//
//void hsvThreshold(cv::Mat &in, cv::Mat &out,
//        const &cv::Scalar min, const &cv::Scalar max) {
//
//#ifdef NOIMP_OAT_USE_CUDA
//    hsv_lut->transform(hsv_image_, lut_frame);
//
//    std::vector<cv::cuda::GpuMat> channels;
//    cv::cuda::split(lut_frame, channels);
//    cv::cuda::bitwise_and(channels[0], channels[1], threshold_frame_);
//    cv::cuda::bitwise_and(channels[2], threshold_frame_, threshold_frame_);
//#endif
//}

//void HSVDetector::erodeDilate(cv::Mat &frame_io, cv::Mat &eroder, cv::Mat &dilator) {
//
//#ifdef NOIMP_OAT_USE_CUDA
//    if (erode_on_)
//        erode_filter->apply(frame_io, frame_io);
//
//    if (dilate_on_)
//        dilate_filter->apply(frame_io, frame_io);
//#endif
//}
//
//
//void HSVDetector::createHSVLUT() {
//
//    std::vector<cv::Mat> lut_channels;
//
//    lut_channels.push_back(cv::Mat(256, 1, CV_8UC1, cv::Scalar(0)));
//    if (h_min < h_max) {
//        auto h_inc = lut_channels[0].rowRange(h_min, h_max);
//        h_inc.setTo(cv::Scalar(1));
//    }
//
//    lut_channels.push_back(cv::Mat(256, 1, CV_8UC1, cv::Scalar(0)));
//    if (s_min < s_max) {
//        auto s_inc = lut_channels[1].rowRange(s_min, s_max);
//        s_inc.setTo(cv::Scalar(1));
//    }
//
//    lut_channels.push_back(cv::Mat(256, 1, CV_8UC1, cv::Scalar(0)));
//    if (v_min < v_max) {
//        auto v_inc = lut_channels[2].rowRange(v_min, v_max);
//        v_inc.setTo(cv::Scalar(1));
//    }
//
//    cv::Mat lut;
//    cv::merge(lut_channels, lut);
//    hsv_lut = cv::cuda::createLookUpTable(lut);
//
//}
//
//void HSVDetector::set_erode_size(int value) {
//
//    if (value > 0) {
//        erode_on = true;
//        erode_px = value;
//        cv::Mat erode_element =
//            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(erode_px, erode_px));
//        erode_filter =
//            cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, erode_element);
//    } else {
//        erode_on = false;
//    }
//}
//
//void HSVDetector::set_dilate_size(int value) {
//    if (value > 0) {
//        dilate_on = true;
//        dilate_px = value;
//        cv::Mat dilate_element =
//            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilate_px, dilate_px));
//        dilate_filter =
//            cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, dilate_element);
//    } else {
//        dilate_on = false;
//    }
//}
//
//#endif // NOIMP_OAT_USE_CUDA
