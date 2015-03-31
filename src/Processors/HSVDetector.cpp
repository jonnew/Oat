/* 
 * File:   HSVFilter.cpp
 * Author: Jon Newman <jpnewman snail mit dot edu>
 * 
 * Created on March 25, 2015, 5:11 PM
 */

#include "HSVDetector.h"

#include <string>
#include <iostream>
#include <limits>
#include <math.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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

HSVDetector::HSVDetector(const std::string filter_name_in,
                     int h_min_in, int h_max_in, 
                     int s_min_in, int s_max_in, 
                     int v_min_in, int v_max_in) {
    
    filter_name = filter_name_in;
    title = filter_name + "_detector";

    // Initial threshold values
    h_min = h_min_in;
    h_max = h_max_in;
    s_min = s_min_in;
    s_max = s_max_in;
    v_min = v_min_in;
    v_max = v_max_in;
    
    // Set defaults for the erode and dilate blocks
    erode_size = Size(2, 2);
    dilate_size = Size(10, 10);
    
    // Relative position of the object
    position = "unknown";
    
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

HSVDetector::HSVDetector(const std::string filter_name_in) : 
    HSVDetector::HSVDetector(filter_name_in, 0, 256, 0, 256, 0, 256) { }

HSVDetector::~HSVDetector() {
}

void HSVDetector::createTrackbars() {

    // Create window for trackbars
    namedWindow(filter_name + "HSV_slider", 0);

    // Create trackbars and insert them into window
    createTrackbar("H_MIN", filter_name, &h_min, h_max, 0);
    createTrackbar("H_MAX", filter_name, &h_max, h_max, 0);
    createTrackbar("S_MIN", filter_name, &s_min, s_max, 0);
    createTrackbar("S_MAX", filter_name, &s_max, s_max, 0);
    createTrackbar("V_MIN", filter_name, &v_min, v_max, 0);
    createTrackbar("V_MAX", filter_name, &v_max, v_max, 0);

}

void HSVDetector::applyFilter(const Mat& rgb_img, Mat& threshold_img) {

    rgb_img.copyTo(image);
    hsvTransform();
    applyThreshold(threshold_img);
    clarifyObjects(threshold_img);
    findObjects(threshold_img);
    decorateFeed(image, cv::Scalar(0, 0, 255));
    
    if (show) {
        showImage();
    }
}

void HSVDetector::hsvTransform() {

    cvtColor(image, image, COLOR_BGR2HSV);
}

void HSVDetector::applyThreshold(Mat& threshold_img) {

    inRange(image, Scalar(h_min, s_min, v_min), Scalar(h_max, s_max, v_max), threshold_img);

}

void HSVDetector::clarifyObjects(Mat& threshold_img) {

    erode_element = getStructuringElement(MORPH_RECT, erode_size);
    dilate_element = getStructuringElement(MORPH_RECT, dilate_size);

    cv::erode(threshold_img, threshold_img, erode_element);
    cv::erode(threshold_img, threshold_img, erode_element);

    cv::dilate(threshold_img, threshold_img, dilate_element);
    cv::dilate(threshold_img, threshold_img, dilate_element);
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

                cv::Moments moment = moments((cv::Mat)contours[index]);
                double area = moment.m00;

                // Isolate the largest contour within the min/max range.
                if (area > min_object_area && area < max_object_area && area > object_area) {
                    xy_coord_px.x = moment.m10 / area;
                    xy_coord_px.y = moment.m01 / area;
                    object_found = true;
                    object_area = area;
                }
            }
        }
        else {
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
        int rad = sqrt(object_area/PI);
        cv::circle(display_img, xy_coord_px, rad, color, 2);
    }
    else {
        cv::putText(display_img, status_text, cv::Point(0, 50), 2, 1, cv::Scalar(255, 255, 255), 2);
    }
    
    
    
}