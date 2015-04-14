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

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "../../lib/cpptoml/cpptoml.h"
#include "../../lib/shmem/MatClient.h"
#include "../../lib/shmem/MatServer.h"
#include "../../lib/shmem/SMServer.h"

using cv::Mat;
using cv::namedWindow;
using cv::createTrackbar;
using cv::cvtColor;
using cv::inRange;
using cv::COLOR_BGR2HSV;
using cv::MORPH_RECT;
using cv::Scalar;
using cv::getStructuringElement;
using cv::Size;
using cv::erode;
using cv::dilate;

HSVDetector::HSVDetector(std::string source_name, std::string pos_sink_name,
        int h_min_in, int h_max_in,
        int s_min_in, int s_max_in,
        int v_min_in, int v_max_in) :
  frame_source(source_name)
, position_sink(pos_sink_name)
, frame_sink(pos_sink_name + "_frame")
, frame_sink_used(false)
{

    detector_name = source_name + "_hsv_detector";
    slider_title = detector_name + "_hsv_sliders";

    // Initial threshold values
    h_min = h_min_in;
    h_max = h_max_in;
    s_min = s_min_in;
    s_max = s_max_in;
    v_min = v_min_in;
    v_max = v_max_in;

    // Set defaults for the erode and dilate blocks
    set_erode_size(0);
    set_dilate_size(10);

    // Relative position of the object
    position = "unknown";

    // Decorate stream
    decorate = false;

    // Initialize area parameters without constraint
    min_object_area = 0;
    max_object_area = std::numeric_limits<double>::max();

    // Maximum number of contours defining candidate objects
    max_num_contours = 50;

    // Initial point is unknown
    object_found = false;
    xy_coord_px.x = 0;
    xy_coord_px.y = 0;
}

HSVDetector::HSVDetector(std::string source_name, std::string pos_sink_name) :
HSVDetector::HSVDetector(source_name, pos_sink_name, 0, 256, 0, 256, 0, 256) { }

void HSVDetector::createTrackbars() {

    // Create window for trackbars
    namedWindow(slider_title, cv::WINDOW_AUTOSIZE);

    // Create trackbars and insert them into window
    // TODO: Does not work for multiple detectors. For the second one, 'this' points to nonsense...
    createTrackbar("H_MIN", slider_title, &h_min, 256); //, &HSVDetector::hminSliderChangedCallback, this);
    createTrackbar("H_MAX", slider_title, &h_max, 256); //, &HSVDetector::hmaxSliderChangedCallback, this);
    createTrackbar("S_MIN", slider_title, &s_min, 256); //, &HSVDetector::sminSliderChangedCallback, this);
    createTrackbar("S_MAX", slider_title, &s_max, 256); //, &HSVDetector::smaxSliderChangedCallback, this);
    createTrackbar("V_MIN", slider_title, &v_min, 256); //, &HSVDetector::vminSliderChangedCallback, this);
    createTrackbar("V_MAX", slider_title, &v_max, 256); //, &HSVDetector::vmaxSliderChangedCallback, this);
    createTrackbar("ERODE", slider_title, &erode_px, 50, &HSVDetector::erodeSliderChangedCallback, this);
    createTrackbar("DILATE", slider_title, &dilate_px, 50, &HSVDetector::dilateSliderChangedCallback, this);

}

void HSVDetector::erodeSliderChangedCallback(int value, void* object) {
    HSVDetector* hsv_detector = (HSVDetector*) object;
    hsv_detector->set_erode_size(value);
}

void HSVDetector::dilateSliderChangedCallback(int value, void* object) {
    HSVDetector* hsv_detector = (HSVDetector*) object;
    hsv_detector->set_dilate_size(value);
}

void HSVDetector::applyFilterAndServe() { 

    proc_mat = frame_source.get_shared_mat().clone();

    hsvTransform(proc_mat);
    applyThreshold(proc_mat, threshold_img);
    applyThresholdMask(threshold_img, proc_mat);
    clarifyObjects(threshold_img);
    findObjects(threshold_img);

    if (decorate) {
        decorateFeed(threshold_img, cv::Scalar(0, 0, 255)); // TODO: get rid of this, replace with position server. we can have a viewer to make things pretty.
    }

    // Put processed mat in shared memory
    if (frame_sink_used) {
        frame_sink.set_shared_mat(threshold_img);
    }
    
    // Put position in shared memory
    position_sink.set_value(xy_coord_px);
    
    frame_source.wait();

}

void HSVDetector::hsvTransform(cv::Mat& rgb_img) {

    cvtColor(rgb_img, rgb_img, COLOR_BGR2HSV);
}

void HSVDetector::applyThreshold(const cv::Mat hsv_img, Mat& threshold_img) {

    inRange(hsv_img, Scalar(h_min, s_min, v_min), Scalar(h_max, s_max, v_max), threshold_img);

}

void HSVDetector::applyThresholdMask(const cv::Mat threshold_img, cv::Mat& hsv_img) {

    hsv_img.setTo(0, threshold_img == 0);
}

void HSVDetector::clarifyObjects(Mat& threshold_img) {

    if (erode_on) {
        cv::erode(threshold_img, threshold_img, erode_element);
        //cv::erode(threshold_img, threshold_img, erode_element);
    }


    if (dilate_on) {
        cv::dilate(threshold_img, threshold_img, dilate_element);
        //cv::dilate(threshold_img, threshold_img, dilate_element);
    }

}

bool HSVDetector::findObjects(const cv::Mat& threshold_img) {

    cv::Mat thesh_cpy = threshold_img.clone();
    std::vector< std::vector < cv::Point > > contours;
    std::vector< cv::Vec4i > hierarchy;

    // This function will modify the threshold_img data.
    cv::findContours(thesh_cpy, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

    object_area = 0;
    object_found = false;

    if (int num_contours = hierarchy.size() > 0) {

        if (num_contours < max_num_contours) {

            for (int index = 0; index >= 0; index = hierarchy[index][0]) {

                cv::Moments moment = cv::moments((cv::Mat)contours[index]);
                double area = moment.m00;

                // Isolate the largest contour within the min/max range.
                if (area > min_object_area && area < max_object_area && area > object_area) {
                    xy_coord_px.x = moment.m10 / area;
                    xy_coord_px.y = moment.m01 / area;
                    object_found = true;
                    object_area = area;
                }
            }
        } else {
            // Issue warning because we found too many contours
            status_text = "Too many contours. Tracking off.";

            //std::cerr << "WARNING: Call to findObjects found more than the maximum allowed number of contours. " << std::endl;
            //std::cerr << "Threshold image too noisy." << std::endl;
        }

    } else {
        // Issue warning because we found no countours
        status_text = "No contours. Tracking off.";
    }

    return object_found;
}

void HSVDetector::decorateFeed(cv::Mat& display_img, const cv::Scalar& color) { //const cv::Scalar& color

    // Add an image of the 
    if (object_found) {

        // Get the radius of the object
        int rad = sqrt(object_area / PI);
        cv::circle(display_img, xy_coord_px, rad, color, 2);
    } else {
        cv::putText(display_img, status_text, cv::Point(5, 35), 2, 1, cv::Scalar(255, 255, 255), 2);
    }
}

void HSVDetector::configure(std::string config_file, std::string key) {

    cpptoml::table config;

    try {
        config = cpptoml::parse_file(config_file);

        //std::cout << "HSVDetector parsed the following configuration..." << std::endl << std::endl;
        //std::cout << config << std::endl;
    } catch (const cpptoml::parse_exception& e) {
        std::cerr << "Failed to parse " << config_file << ": " << e.what() << std::endl;
    }

    try {
        // See if a camera configuration was provided
        if (config.contains(key)) {

            auto hsv_config = *config.get_table(key);
            set_detector_name(key);

            if (hsv_config.contains("position")) {
                position = *hsv_config.get_as<std::string>("position");
            }

            if (hsv_config.contains("decorate")) {
                decorate = *hsv_config.get_as<bool>("decorate");
            }

            if (hsv_config.contains("erode")) {
                set_erode_size((int) (*hsv_config.get_as<int64_t>("erode")));
            }

            if (hsv_config.contains("dilate")) {
                set_dilate_size((int) (*hsv_config.get_as<int64_t>("dilate")));
            }

            if (hsv_config.contains("h_thresholds")) {
                auto t = *hsv_config.get_table("h_thresholds");

                if (t.contains("min")) {
                    h_min = (int) (*t.get_as<int64_t>("min"));
                }
                if (t.contains("max")) {
                    h_max = (int) (*t.get_as<int64_t>("max"));
                }
            }

            if (hsv_config.contains("s_thresholds")) {
                auto t = *hsv_config.get_table("s_thresholds");

                if (t.contains("min")) {
                    s_min = (int) (*t.get_as<int64_t>("min"));
                }
                if (t.contains("max")) {
                    s_max = (int) (*t.get_as<int64_t>("max"));
                }
            }

            if (hsv_config.contains("v_thresholds")) {
                auto t = *hsv_config.get_table("v_thresholds");

                if (t.contains("min")) {
                    v_min = (int) (*t.get_as<int64_t>("min"));
                }
                if (t.contains("max")) {
                    v_max = (int) (*t.get_as<int64_t>("max"));
                }
            }

            if (hsv_config.contains("hsv_tune")) {
                if (*hsv_config.get_as<bool>("hsv_tune"))
                    createTrackbars();
            }

        } else {
            std::cerr << "No HSV detector configuration named \"" + key + "\" was provided. Exiting." << std::endl;
            exit(EXIT_FAILURE);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}

void HSVDetector::addFrameSink(std::string frame_sink_name) {
    frame_sink.set_name(frame_sink_name);
    frame_sink_used = true;
}

void HSVDetector::set_erode_size(int _erode_px) {

    if (_erode_px > 0) {
        erode_on = true;
        erode_px = _erode_px;
        erode_element = getStructuringElement(MORPH_RECT, cv::Size(erode_px, erode_px));
    } else {
        erode_on = false;
    }
}

void HSVDetector::set_dilate_size(int _dilate_px) {
    if (_dilate_px > 0) {
        dilate_on = true;
        dilate_px = _dilate_px;
        dilate_element = getStructuringElement(MORPH_RECT, cv::Size(dilate_px, dilate_px));
    } else {
        dilate_on = false;
    }
}