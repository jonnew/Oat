/* 
 * File:   HSVFilter.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on March 25, 2015, 5:11 PM
 */

#ifndef HSVFILTER_H
#define	HSVFILTER_H

#include <string>
#include <opencv2/core/mat.hpp>

class HSVFilter {
    
public:
    
    // Manual tuning of HSV values using graphical track bars
    HSVFilter(std::string filter_name);
    
    // HSV values known
    HSVFilter(std::string filter_name,
              int h_min, int h_max, 
              int s_min, int s_max,
              int v_min, int v_max);
    
    virtual ~HSVFilter(void);
    
    // Apply the HSVTransform, thresholding, and erode/dilate operations
    void applyFilter(const cv::Mat& rgb_img, cv::Mat& threshold_img);

    // TODO: These will need to be tuned to the actual application
    void set_h_min(int h_min) { h_min = h_min;}
    void set_h_max(int h_max) { h_max = h_max;}
    void set_s_min(int s_min) { s_min = s_min;}
    void set_s_max(int s_max) { s_max = s_max;}
    void set_v_min(int v_min) { v_min = v_min;}
    void set_v_max(int v_max) { v_max = v_max;}
    void set_erode_size(cv::Size erode_size) { erode_size = erode_size;}
    void set_dilate_size(cv::Size dilate_size) { dilate_size = dilate_size;}

private:

    // Internal data images
    cv::Mat hsv_img;
    
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
    void createTrackbars(void);
    
    // HSV filter
    void hsvTransform(void);

    // Binary threshold
    void applyThreshold(cv::Mat& threshold_img);
    
    // Erode/dilate objects to get rid of speckles
    void clarifyObjects(cv::Mat& threshold_img);

};

#endif	/* HSVFILTER_H */

