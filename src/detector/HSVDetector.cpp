//******************************************************************************
//* Copyright (c) Jon Newman (jpnewman at mit snail edu) 
//* All right reserved.
//* This file is part of the Simple Tracker project.
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

#include "HSVDetector.h"

#include <string>
#include <iostream>
#include <limits>
#include <math.h>
#include <opencv2/opencv.hpp>

#include "../../lib/cpptoml/cpptoml.h"
#include "../../lib/shmem/MatServer.h"

HSVDetector::HSVDetector(std::string image_source_name, std::string position_sink_name,
        int h_min_in, int h_max_in,
        int s_min_in, int s_max_in,
        int v_min_in, int v_max_in) :
  Detector(image_source_name, position_sink_name)
//, frame_sink(position_sink_name + "_frame")
//, frame_sink_used(false)
, h_min(h_min_in)
, h_max(h_max_in)
, s_min(s_min_in)
, s_max(s_max_in)
, v_min(v_min_in)
, v_max(v_max_in)
, name(position_sink_name + "_hsv") {

    tuning_on = false;

    // Set defaults for the erode and dilate blocks
    set_erode_size(0);
    set_dilate_size(10);

    // Initialize area parameters without constraint
    min_object_area = 0;
    max_object_area = std::numeric_limits<double>::max(); //TODO: These are not being used and would be helpful, esp max_area!
}

HSVDetector::HSVDetector(std::string source_name, std::string pos_sink_name) :
HSVDetector::HSVDetector(source_name, pos_sink_name, 0, 256, 0, 256, 0, 256) {
}

void HSVDetector::erodeSliderChangedCallback(int value, void* object) {
    HSVDetector* hsv_detector = (HSVDetector*) object;
    hsv_detector->set_erode_size(value);
}

void HSVDetector::dilateSliderChangedCallback(int value, void* object) {
    HSVDetector* hsv_detector = (HSVDetector*) object;
    hsv_detector->set_dilate_size(value);
}

void HSVDetector::findObject() {

    hsv_image = image_source.get_value().clone();

    cv::cvtColor(hsv_image, hsv_image, cv::COLOR_BGR2HSV);
    applyThreshold();
    clarifyBlobs();
    siftBlobs();
    tune();
}

void HSVDetector::servePosition() {
    // Put position in shared memory
    position_sink.set_value(object_position);
}

void HSVDetector::applyThreshold() {

    cv::inRange(hsv_image, cv::Scalar(h_min, s_min, v_min), cv::Scalar(h_max, s_max, v_max), threshold_image);
    hsv_image.setTo(0, threshold_image == 0);

}

void HSVDetector::clarifyBlobs() {

    if (erode_on) {
        cv::erode(threshold_image, threshold_image, erode_element);
    }

    if (dilate_on) {
        cv::dilate(threshold_image, threshold_image, dilate_element);
    }

}

void HSVDetector::siftBlobs() {

    cv::Mat thesh_cpy = threshold_image.clone();
    std::vector< std::vector < cv::Point > > contours;
    std::vector< cv::Vec4i > hierarchy;

    // This function will modify the threshold_img data.
    cv::findContours(thesh_cpy, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

    object_area = 0;
    object_position.position_valid = false;

    if (hierarchy.size() > 0) {

        for (int index = 0; index >= 0; index = hierarchy[index][0]) {

            cv::Moments moment = cv::moments((cv::Mat)contours[index]);
            double area = moment.m00;

            // Isolate the largest contour within the min/max range.
            if (area > min_object_area && area < max_object_area && area > object_area) {
                object_position.position.x = moment.m10 / area;
                object_position.position.y = moment.m01 / area;
                object_position.position_valid = true;
                object_area = area;
            }
        }
    }
}

//void HSVDetector::decorateFeed(cv::Mat& display_img, const cv::Scalar& color) { //const cv::Scalar& color
//
//    // Add an image of the 
//    if (object_position.position_valid) {
//
//        // Get the radius of the object
//        int rad = sqrt(object_area / PI);
//        cv::circle(display_img, object_position.position, rad, color, 2);
//    } else {
//        cv::putText(display_img, status_text, cv::Point(5, 35), 2, 1, cv::Scalar(255, 255, 255), 2);
//    }
//}

void HSVDetector::configure(std::string file_name, std::string key) {

    cpptoml::table config;

    try {
        config = cpptoml::parse_file(file_name);
    } catch (const cpptoml::parse_exception& e) {
        std::cerr << "Failed to parse " << file_name << ": " << e.what() << std::endl;
    }

    try {
        // See if a camera configuration was provided
        if (config.contains(key)) {

            auto this_config = *config.get_table(key);

            if (this_config.contains("erode")) {
                set_erode_size((int) (*this_config.get_as<int64_t>("erode")));
            }

            if (this_config.contains("dilate")) {
                set_dilate_size((int) (*this_config.get_as<int64_t>("dilate")));
            }

            if (this_config.contains("h_thresholds")) {
                auto t = *this_config.get_table("h_thresholds");

                if (t.contains("min")) {
                    h_min = (int) (*t.get_as<int64_t>("min"));
                }
                if (t.contains("max")) {
                    h_max = (int) (*t.get_as<int64_t>("max"));
                }
            }

            if (this_config.contains("s_thresholds")) {
                auto t = *this_config.get_table("s_thresholds");

                if (t.contains("min")) {
                    s_min = (int) (*t.get_as<int64_t>("min"));
                }
                if (t.contains("max")) {
                    s_max = (int) (*t.get_as<int64_t>("max"));
                }
            }

            if (this_config.contains("v_thresholds")) {
                auto t = *this_config.get_table("v_thresholds");

                if (t.contains("min")) {
                    v_min = (int) (*t.get_as<int64_t>("min"));
                }
                if (t.contains("max")) {
                    v_max = (int) (*t.get_as<int64_t>("max"));
                }
            }

            if (this_config.contains("tune")) {
                if (*this_config.get_as<bool>("tune")) {
                    tuning_on = true;
                    createTuningWindows();
                }
            }

        } else {
            std::cerr << "No HSV detector configuration named \"" + key + "\" was provided. Exiting." << std::endl;
            exit(EXIT_FAILURE);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}

void HSVDetector::tune() {

    if (tuning_on) {
        if (!tuning_windows_created) {
            createTuningWindows();
        }
        cv::imshow(tuning_image_title, threshold_image);
        cv::waitKey(1);
    } else if (!tuning_on && tuning_windows_created) {
        // Destroy the tuning windows
        cv::destroyWindow(tuning_image_title);
        cv::destroyWindow(slider_title);
        tuning_windows_created = false;
    }
}

void HSVDetector::createTuningWindows() {

    // Create window for sliders
    cv::namedWindow(slider_title, cv::WINDOW_AUTOSIZE);

    // Create sliders and insert them into window
    cv::createTrackbar("H_MIN", slider_title, &h_min, 256); 
    cv::createTrackbar("H_MAX", slider_title, &h_max, 256); 
    cv::createTrackbar("S_MIN", slider_title, &s_min, 256); 
    cv::createTrackbar("S_MAX", slider_title, &s_max, 256); 
    cv::createTrackbar("V_MIN", slider_title, &v_min, 256); 
    cv::createTrackbar("V_MAX", slider_title, &v_max, 256); 
    cv::createTrackbar("ERODE", slider_title, &erode_px, 50, &HSVDetector::erodeSliderChangedCallback, this);
    cv::createTrackbar("DILATE", slider_title, &dilate_px, 50, &HSVDetector::dilateSliderChangedCallback, this);
    
    tuning_windows_created = true;
}

void HSVDetector::set_erode_size(int value) {

    if (value > 0) {
        erode_on = true;
        erode_px = value;
        erode_element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(erode_px, erode_px));
    } else {
        erode_on = false;
    }
}

void HSVDetector::set_dilate_size(int value) {
    if (value > 0) {
        dilate_on = true;
        dilate_px = value;
        dilate_element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilate_px, dilate_px));
    } else {
        dilate_on = false;
    }
}