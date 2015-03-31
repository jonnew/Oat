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
    slider_title = filter_name + "_hsv_sliders";

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
    namedWindow(slider_title, cv::WINDOW_AUTOSIZE);

    // Create trackbars and insert them into window
    // TODO: Does not work for multiple detectors. For the second one, 'this' points to nonsense...
    createTrackbar("H_MIN", slider_title, &h_min, 256, &HSVDetector::hminSliderChangedCallback, this);
    createTrackbar("H_MAX", slider_title, &h_max, 256, &HSVDetector::hmaxSliderChangedCallback, this);
    createTrackbar("S_MIN", slider_title, &s_min, 256, &HSVDetector::sminSliderChangedCallback, this);
    createTrackbar("S_MAX", slider_title, &s_max, 256, &HSVDetector::smaxSliderChangedCallback, this);
    createTrackbar("V_MIN", slider_title, &v_min, 256, &HSVDetector::vminSliderChangedCallback, this);
    createTrackbar("V_MAX", slider_title, &v_max, 256, &HSVDetector::vmaxSliderChangedCallback, this);

}

void HSVDetector::hminSliderChangedCallback(int value, void* object) {
    HSVDetector* hsv_detector = (HSVDetector*) object;
    hsv_detector->h_min = value;
}

void HSVDetector::hmaxSliderChangedCallback(int value, void* object) {
    HSVDetector* hsv_detector = (HSVDetector*) object;
    hsv_detector->h_max = value;
}

void HSVDetector::sminSliderChangedCallback(int value, void* object) {
    HSVDetector* hsv_detector = (HSVDetector*) object;
    hsv_detector->s_min = value;
}

void HSVDetector::smaxSliderChangedCallback(int value, void* object) {
    HSVDetector* hsv_detector = (HSVDetector*) object;
    hsv_detector->s_max = value;
}

void HSVDetector::vminSliderChangedCallback(int value, void* object) {
    HSVDetector* hsv_detector = (HSVDetector*) object;
    hsv_detector->v_min = value;
}

void HSVDetector::vmaxSliderChangedCallback(int value, void* object) {
    HSVDetector* hsv_detector = (HSVDetector*) object;
    hsv_detector->v_max = value;
}

void HSVDetector::applyFilter(const Mat& rgb_img, Mat& threshold_img) {

    rgb_img.copyTo(image);
    hsvTransform();
    applyThreshold(threshold_img);
    applyThresholdMask(threshold_img);
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

void HSVDetector::applyThresholdMask(Mat& threshold_img) {
    
    image.setTo(0,threshold_img == 0);
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
        cv::putText(display_img, status_text, cv::Point(5, 35), 2, 1, cv::Scalar(255, 255, 255), 2);
    }
    
    
    
}