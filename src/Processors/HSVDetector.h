/* 
 * File:   HSVFilter.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on March 25, 2015, 5:11 PM
 */

#ifndef HSVFILTER_H
#define	HSVFILTER_H

#include "ImageDisplay.h"
#include <string>
#include <opencv2/core/mat.hpp>

#define PI 3.14159265358979323846

class Tracker;

class HSVDetector : public ImageDisplay {
    friend Tracker;

public:
    
    // HSV values known
    HSVDetector(const std::string filter_name,
            int h_min, int h_max,
            int s_min, int s_max,
            int v_min, int v_max);
    
    // Start with full range of HSV thresholds. This is typically used
    // for manual tuning of thresholds using the createTrackbars call
    HSVDetector(const std::string filter_name);

    virtual ~HSVDetector(void);

    // Sliders to allow manipulation of HSV thresholds
    void createTrackbars(void);

    // Object detection
    bool findObjects(const cv::Mat& threshold_img);

    // Apply the HSVTransform, thresholding, and erode/dilate operations
    void applyFilter(const cv::Mat& rgb_img, cv::Mat& threshold_img);

    // Accessors

    void set_max_num_contours(unsigned int max_num_contours) {
        max_num_contours = max_num_contours;
    }

    void set_min_object_area(double min_object_area) {
        min_object_area = min_object_area;
    }

    void set_max_object_area(double max_object_area) {
        max_object_area = max_object_area;
    }

    void set_h_min(int h_min) {
        h_min = h_min;
    }

    void set_h_max(int h_max) {
        h_max = h_max;
    }

    void set_s_min(int s_min) {
        s_min = s_min;
    }

    void set_s_max(int s_max) {
        s_max = s_max;
    }

    void set_v_min(int v_min) {
        v_min = v_min;
    }

    void set_v_max(int v_max) {
        v_max = v_max;
    }

    void set_erode_size(int erode_px) {
        erode_size = cv::Size(erode_px, erode_px);
    }

    void set_dilate_size(int dilate_px) {
        dilate_size = cv::Size(dilate_px, dilate_px);
    }

private:

    // Sizes of the erode and dilate blocks
    cv::Size erode_size, dilate_size;
    cv::Mat erode_element, dilate_element;

    // Initial threshold values
    int h_min;
    int h_max;
    int s_min;
    int s_max;
    int v_min;
    int v_max;

    // For manual manipulation of HSV filtering
    std::string filter_name;

    // Object detection parameters
    std::string position;
    float mm_per_px;

    std::string status_text;
    bool object_found;
    double object_area;
    cv::Point xy_coord_px;
    cv::Point xy_coord_mm;

    unsigned int max_num_contours;
    double min_object_area;
    double max_object_area;

    // HSV filter
    void hsvTransform(void);

    // Binary threshold
    void applyThreshold(cv::Mat& threshold_img);

    // Erode/dilate objects to get rid of speckles
    void clarifyObjects(cv::Mat& threshold_img);

    // Add information to displayed image
    void decorateFeed(cv::Mat& display_img, const cv::Scalar&);

};

#endif	/* HSVFILTER_H */

