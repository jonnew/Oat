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


#include "DifferenceDetector.h"

#include <string>
#include <opencv2/opencv.hpp>

#include "../../lib/cpptoml/cpptoml.h"

DifferenceDetector::DifferenceDetector(std::string image_source_name, std::string position_sink_name) :
  Detector(image_source_name, position_sink_name)
, last_image_set(false) { 
    
    set_blur_size(2);
}

void DifferenceDetector::servePosition() {
    position_sink.pushObject(object_position);
}

void DifferenceDetector::findObject() {

    this_image = image_source.get_value().clone();
    applyThreshold();
    siftBlobs();
    tune();  
    
}

void DifferenceDetector::configure(std::string file_name, std::string key) {

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

void DifferenceDetector::siftBlobs() {

    cv::Mat thresh_cpy = threshold_image.clone();
    std::vector< std::vector < cv::Point > > contours;
    std::vector< cv::Vec4i > hierarchy;
    
    //these two vectors needed for output of findContours
    //find contours of filtered image using openCV findContours function
    //findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );// retrieves all contours
    cv::findContours(thresh_cpy, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE); // retrieves external contours

    //if contours vector is not empty, we have found some objects
    if (contours.size() > 0) {
        object_position.position_valid = true;
    }
    else 
        object_position.position_valid = false;

    if (object_position.position_valid) {
        
        //the largest contour is found at the end of the contours vector
        //we will simply assume that the biggest contour is the object we are looking for.
        std::vector< std::vector<cv::Point> > largestContourVec;
        largestContourVec.push_back(contours.at(contours.size() - 1));
        
        //make a bounding rectangle around the largest contour then find its centroid
        //this will be the object's final estimated position.
        cv::Rect objectBoundingRectangle = cv::boundingRect(largestContourVec.at(0));
        object_position.position.x = objectBoundingRectangle.x + 0.5 * objectBoundingRectangle.width;
        object_position.position.y = objectBoundingRectangle.y + 0.5 * objectBoundingRectangle.height;
        
        if (tuning_on) {
            cv::cvtColor(threshold_image, threshold_image, cv::COLOR_GRAY2BGR);
            cv::rectangle( threshold_image, objectBoundingRectangle.tl(), objectBoundingRectangle.br(), cv::Scalar(0, 0, 255), 2);
        }
    }
}

void DifferenceDetector::applyThreshold() {

    if (last_image_set){
        cv::cvtColor(this_image, this_image, cv::COLOR_BGR2GRAY);
        cv::absdiff(this_image, last_image, threshold_image);
        cv::threshold(threshold_image, threshold_image, difference_intensity_threshold, 255, cv::THRESH_BINARY);
        if (blur_on){
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

void DifferenceDetector::tune() {

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

void DifferenceDetector::createTuningWindows() {
    
    //cv::startWindowThread();
    
    // Create window for sliders
    cv::namedWindow(tuning_image_title);
    cv::namedWindow(slider_title, cv::WINDOW_NORMAL);

    // Create sliders and insert them into window
    cv::createTrackbar("THRESH", slider_title, &difference_intensity_threshold, 256); 
    cv::createTrackbar("BLUR", slider_title, &blur_size.height, 50, &DifferenceDetector::blurSliderChangedCallback, this);
    
    tuning_windows_created = true;
}

void DifferenceDetector::blurSliderChangedCallback(int value, void* object) {
    DifferenceDetector* diff_detector = (DifferenceDetector*) object;
    diff_detector->set_blur_size(value);
}

void DifferenceDetector::set_blur_size(int value) {

    if (value > 0) {
        blur_on = true;
        blur_size = cv::Size(value, value);
    } else {
        blur_on = false;
    }
}