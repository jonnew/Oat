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


DifferenceDetector::DifferenceDetector(std::string image_source_name, std::string position_sink_name) :
  image_source(image_source_name)
, position_sink(position_sink_name)
, last_image_set(false)
, slider_title(position_sink_name + "_sliders") { }

DifferenceDetector::DifferenceDetector(const DifferenceDetector& orig) {
}

DifferenceDetector::~DifferenceDetector() {
}

void DifferenceDetector::servePosition() {
    position_sink.set_value(object_position);
}


void DifferenceDetector::findObject() {

    cv::Mat thresh_cpy = threshold_image.clone();
    std::vector< std::vector < cv::Point > > contours;
    std::vector< cv::Vec4i > hierarchy;
    
    //these two vectors needed for output of findContours
    //find contours of filtered image using openCV findContours function
    //findContours(temp,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE );// retrieves all contours
    cv::findContours(thresh_cpy, contours, hierarchy, cv::CV_RETR_EXTERNAL, cv::CV_CHAIN_APPROX_SIMPLE); // retrieves external contours

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
    }
}

void DifferenceDetector::thresholdImage() {

    cv::Mat this_image = image_source.get_value().clone();
    
    if (last_image_set){
        cv::cvtColor(this_image, this_image, cv::COLOR_BGR2GRAY);
        cv::absdiff(this_image, last_image, threshold_image);
        cv::threshold(threshold_image, threshold_image, threshold_level, 255, cv::THRESH_BINARY);
        cv::blur(threshold_image, threshold_image, cv::Size(blur_size, blur_size));
        cv::threshold(threshold_image, threshold_image, threshold_level, 255, cv::THRESH_BINARY);
    }
    
    // Get a copy of the last image
    last_image = this_image.clone();
    last_image_set = true;
}


void DifferenceDetector::createSliders() {
    
    // Create window for sliders
    namedWindow(slider_title, cv::WINDOW_AUTOSIZE);

    // Create sliders and insert them into window
    createTrackbar("THRESH", slider_title, &threshold_level, 256); 
    createTrackbar("BLUR", slider_title, &blur_size, 50, &DifferenceDetector::blurSliderChangedCallback, this);
}

void DifferenceDetector::blurSliderChangedCallback(int value, void* object) {
    DifferenceDetector* diff_detector = (DifferenceDetector*) object;
    diff_detector->set_blur_size(value);
}