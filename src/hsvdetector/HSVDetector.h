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
#include "../../lib/shmem/MatClient.h"
#include "../../lib/shmem/MatServer.h"

#define PI 3.14159265358979323846

class HSVDetector : public MatClient, public MatServer {
    
public:
    
    // HSV values known
    HSVDetector(const std::string source_name, const std::string sink_name,
                int h_min, int h_max,
                int s_min, int s_max,
                int v_min, int v_max);
    
    // Start with full range of HSV thresholds. This is typically used
    // for manual tuning of thresholds using the createTrackbars call
    HSVDetector(const std::string source_name, const std::string sink_name);

    // Use a configuration file to specify parameters
    void configure(std::string config_file, std::string key);
    
    // Sliders to allow manipulation of HSV thresholds
    void createTrackbars(void);

    // Object detection
    bool findObjects(const cv::Mat& threshold_img);

    // Apply the HSVTransform, thresholding, and erode/dilate operations to/from
    // shared memory allocated mat objects
    void applyFilter(void);

    // Accessors
    
    std::string get_detector_name(){
        return detector_name;
    }
    
    void set_detector_name(std::string detector_name) {
        detector_name = detector_name;
        slider_title = detector_name + "_sliders";
    }
    
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
    
    void set_decorate (bool decorate){
        decorate = decorate;
    }

private:

    // Sizes of the erode and dilate blocks
    cv::Size erode_size, dilate_size;
    cv::Mat proc_mat, threshold_img, erode_element, dilate_element;

    // Initial threshold values
    int h_min;
    int h_max;
    int s_min;
    int s_max;
    int v_min;
    int v_max;

    // For manual manipulation of HSV filtering
    std::string detector_name, slider_title;

    // Object detection parameters
    std::string position;
    float mm_per_px;

    std::string status_text;
    bool object_found;
    double object_area;
    cv::Point xy_coord_px;
    cv::Point xy_coord_mm;
    bool decorate; 

    unsigned int max_num_contours;
    double min_object_area;
    double max_object_area;

    // HSV filter
    void hsvTransform(cv::Mat& rgb_img);

    // Binary threshold
    void applyThreshold(const cv::Mat hsv_img, cv::Mat& threshold_img);
    
    // Use the binary threshold to mask the image
    void applyThresholdMask(const cv::Mat threshold_img, cv::Mat& hsv_img);

    // Erode/dilate objects to get rid of speckles
    void clarifyObjects(cv::Mat& threshold_img);

    // Add information to displayed image
    void decorateFeed(cv::Mat& display_img, const cv::Scalar&);
    
    // Callbacks for sliders
    static void hminSliderChangedCallback(int, void*);
    static void hmaxSliderChangedCallback(int, void*);
    static void sminSliderChangedCallback(int, void*);
    static void smaxSliderChangedCallback(int, void*);
    static void vminSliderChangedCallback(int, void*);
    static void vmaxSliderChangedCallback(int, void*);

};

#endif	/* HSVFILTER_H */

