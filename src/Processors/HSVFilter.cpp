/* 
 * File:   HSVFilter.cpp
 * Author: Jon Newman <jpnewman snail mit dot edu>
 * 
 * Created on March 25, 2015, 5:11 PM
 */

#include "HSVFilter.h"

#include <string>
#include <iostream>

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

HSVFilter::HSVFilter(std::string filter_name_in) {

    filter_name = filter_name_in;
    
    // Initial threshold values
    h_min = 0;
    h_max = 256;
    s_min = 0;
    s_max = 256;
    v_min = 0;
    v_max = 256;

    // Tracks for adjusting HSV thresholds
    createTrackbars();

    // Set defaults for the erode and dilate blocks
    erode_size = Size(2, 2);
    dilate_size = Size(10, 10);
}

HSVFilter::HSVFilter(std::string filter_name_in,
                     int h_min_in, int h_max_in, 
                     int s_min_in, int s_max_in, 
                     int v_min_in, int v_max_in) {
    
    filter_name = filter_name_in;

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
}

HSVFilter::~HSVFilter() {
}

void HSVFilter::createTrackbars() {

    // Create window for trackbars
    namedWindow(filter_name + "HSV slider", 0);

    // Create trackbars and insert them into window
    createTrackbar("H_MIN", filter_name, &h_min, h_max, 0);
    createTrackbar("H_MAX", filter_name, &h_max, h_max, 0);
    createTrackbar("S_MIN", filter_name, &s_min, s_max, 0);
    createTrackbar("S_MAX", filter_name, &s_max, s_max, 0);
    createTrackbar("V_MIN", filter_name, &v_min, v_max, 0);
    createTrackbar("V_MAX", filter_name, &v_max, v_max, 0);

}

void HSVFilter::applyFilter(const Mat& rgb_img, Mat& threshold_img) {

    rgb_img.copyTo(hsv_img);
    hsvTransform();
    applyThreshold(threshold_img);
    clarifyObjects(threshold_img);
}

void HSVFilter::hsvTransform() {


    cvtColor(hsv_img, hsv_img, COLOR_BGR2HSV);
    return;
}

void HSVFilter::applyThreshold(Mat& threshold_img) {

    inRange(hsv_img, Scalar(h_min, s_min, v_min), Scalar(h_max, s_max, v_max), threshold_img);

}

void HSVFilter::clarifyObjects(Mat& threshold_img) {

    erode_element = getStructuringElement(MORPH_RECT, erode_size);
    dilate_element = getStructuringElement(MORPH_RECT, dilate_size);

    cv::erode(threshold_img, threshold_img, erode_element);
    cv::erode(threshold_img, threshold_img, erode_element);

    cv::dilate(threshold_img, threshold_img, dilate_element);
    cv::dilate(threshold_img, threshold_img, dilate_element);
}