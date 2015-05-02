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


#include "DifferenceDetector2D.h"

#include <string>
#include <opencv2/opencv.hpp>

#include "../../lib/cpptoml/cpptoml.h"

DifferenceDetector2D::DifferenceDetector2D(std::string image_source_name, std::string position_sink_name) :
Detector2D(image_source_name, position_sink_name)
, last_image_set(false) {

    set_blur_size(2);
}

void DifferenceDetector2D::findObjectAndServePosition() {

    // If we are able to get a an image
    if (image_source.getSharedMat(this_image)) {
        addHomography();
        applyThreshold();
        siftBlobs();
        tune();

        position_sink.pushObject(object_position);
    }
}

void DifferenceDetector2D::configure(std::string file_name, std::string key) {

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

            if (this_config.contains("blur")) {
                set_blur_size((int) (*this_config.get_as<int64_t>("blur")));
            }

            if (this_config.contains("diff_threshold")) {
                difference_intensity_threshold = (int) (*this_config.get_as<int64_t>("diff_threshold"));
            }

            if (this_config.contains("tune")) {
                if (*this_config.get_as<bool>("tune")) {
                    tuning_on = true;
                    createTuningWindows();
                }
            }

        } else {
            std::cerr << "No DifferenceDetector configuration named \"" + key + "\" was provided. Exiting." << std::endl;
            exit(EXIT_FAILURE);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}

void DifferenceDetector2D::siftBlobs() {

    cv::Mat thresh_cpy = threshold_image.clone();
    std::vector< std::vector < cv::Point > > contours;
    std::vector< cv::Vec4i > hierarchy;
    cv::Rect objectBoundingRectangle;

    //these two vectors needed for output of findContours
    //find contours of filtered image using openCV findContours function
    //findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );// retrieves all contours
    cv::findContours(thresh_cpy, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE); // retrieves external contours

    //if contours vector is not empty, we have found some objects
    if (contours.size() > 0) {
        object_position.position_valid = true;
    } else
        object_position.position_valid = false;

    if (object_position.position_valid) {

        //the largest contour is found at the end of the contours vector
        //we will simply assume that the biggest contour is the object we are looking for.
        std::vector< std::vector<cv::Point> > largestContourVec;
        largestContourVec.push_back(contours.at(contours.size() - 1));

        //make a bounding rectangle around the largest contour then find its centroid
        //this will be the object's final estimated position.
        objectBoundingRectangle = cv::boundingRect(largestContourVec.at(0));
        object_position.position.x = objectBoundingRectangle.x + 0.5 * objectBoundingRectangle.width;
        object_position.position.y = objectBoundingRectangle.y + 0.5 * objectBoundingRectangle.height;
    }

    if (tuning_on) {

        std::string msg = cv::format("Object not found"); // TODO: This default msg will not show up. I have no idea why.

        // Plot a circle representing found object
        if (object_position.position_valid) {
            cv::cvtColor(threshold_image, threshold_image, cv::COLOR_GRAY2BGR);
            cv::rectangle(threshold_image, objectBoundingRectangle.tl(), objectBoundingRectangle.br(), cv::Scalar(0, 0, 255), 2);

            // Tell object position
            if (object_position.homography_valid) {
                datatypes::Position2D convert_pos = object_position.convertToWorldCoordinates();
                msg = cv::format("(%.3f, %.3f) world units", convert_pos.position.x, convert_pos.position.y);

            } else {
                msg = cv::format("(%d, %d) pixels", (int) object_position.position.x, (int) object_position.position.y);

            }
        }

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(msg, 1, 1, 1, &baseline);
        cv::Point text_origin(
                threshold_image.cols - textSize.width - 10,
                threshold_image.rows - 2 * baseline - 10);

        cv::putText(threshold_image, msg, text_origin, 1, 1, cv::Scalar(0, 255, 0));
    }
}

void DifferenceDetector2D::applyThreshold() {

    if (last_image_set) {
        cv::cvtColor(this_image, this_image, cv::COLOR_BGR2GRAY);
        cv::absdiff(this_image, last_image, threshold_image);
        cv::threshold(threshold_image, threshold_image, difference_intensity_threshold, 255, cv::THRESH_BINARY);
        if (blur_on) {
            cv::blur(threshold_image, threshold_image, blur_size);
        }
        cv::threshold(threshold_image, threshold_image, difference_intensity_threshold, 255, cv::THRESH_BINARY);
        last_image = this_image.clone(); // Get a copy of the last image
    } else {
        threshold_image = this_image.clone();
        cv::cvtColor(threshold_image, threshold_image, cv::COLOR_BGR2GRAY);
        last_image = this_image.clone();
        cv::cvtColor(last_image, last_image, cv::COLOR_BGR2GRAY);
        last_image_set = true;
    }
}

void DifferenceDetector2D::tune() {

    tuning_mutex.lock();

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

    tuning_mutex.unlock();
}

void DifferenceDetector2D::createTuningWindows() {

    //cv::startWindowThread();

    // Create window for sliders
    cv::namedWindow(tuning_image_title);
    cv::namedWindow(slider_title, cv::WINDOW_NORMAL);

    // Create sliders and insert them into window
    cv::createTrackbar("THRESH", slider_title, &difference_intensity_threshold, 256);
    cv::createTrackbar("BLUR", slider_title, &blur_size.height, 50, &DifferenceDetector2D::blurSliderChangedCallback, this);

    tuning_windows_created = true;
}

void DifferenceDetector2D::blurSliderChangedCallback(int value, void* object) {
    DifferenceDetector2D* diff_detector = (DifferenceDetector2D*) object;
    diff_detector->set_blur_size(value);
}

void DifferenceDetector2D::set_blur_size(int value) {

    if (value > 0) {
        blur_on = true;
        blur_size = cv::Size(value, value);
    } else {
        blur_on = false;
    }
}
