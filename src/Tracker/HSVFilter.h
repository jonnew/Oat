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
    // Manual tuning of HSV values
    HSVFilter(void);
    
    // HSV values known and static
    HSVFilter(int h_min_in, int h_max_in, int s_min_in, int s_max_in, int v_min_in, int v_max_in);
    
    virtual ~HSVFilter(void);
    
    // Apply the HSVTransform, thresholding, and erode/dilate operations
    void applyFilter(const cv::Mat& rgb_img, cv::Mat& threshold_img);

    // TODO: These will need to be tuned to the actual application
    inline void set_erode_size(cv::Size erode_size) { erode_size = erode_size;}
    inline void set_dilate_size(cv::Size dilate_size) { dilate_size = dilate_size;}

   
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
    std::string trackbarWindowName;
    void createTrackbars(void);
    
    // HSV filter
    void hsvTransform(void);

    // Binary threshold
    void applyThreshold(cv::Mat& threshold_img);
    
    // Erode/dilate objects to get rid of speckles
    void clarifyObjects(cv::Mat& threshold_img);


};

#endif	/* HSVFILTER_H */

