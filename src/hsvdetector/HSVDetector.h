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

#ifndef HSVFILTER_H
#define	HSVFILTER_H

#include <string>
#include <opencv2/core/mat.hpp>

#include "../../lib/shmem/MatClient.h"
#include "../../lib/shmem/MatServer.h"
#include "../../lib/shmem/SMServer.h"
#include "../../lib/shmem/Position2D.h"

#define PI 3.14159265358979323846

class HSVDetector {
    
public:
    
    // HSV values known
    HSVDetector(const std::string source_name, const std::string pos_sink_name,
                int h_min, int h_max,
                int s_min, int s_max,
                int v_min, int v_max);
    
    // Start with full range of HSV thresholds. This is typically used
    // for manual tuning of thresholds using the createTrackbars call
    HSVDetector(const std::string source_name, const std::string pos_sink_name);

    // Use a configuration file to specify parameters
    void configure(std::string config_file, std::string key);
    
    // Add a frame sink to view the filtered output. Not normally needed.
    void addFrameSink(std::string frame_sink_name);
    
    // Sliders to allow manipulation of HSV thresholds
    void createTrackbars(void);

    // Object detection
    bool findObjects(const cv::Mat& threshold_img);

    // Apply the HSVTransform, thresholding, and erode/dilate operations to/from
    // shared memory allocated mat objects
    void applyFilterAndServe(void);
    
    // Following filtering, serve position object
    void sendPosition(void);

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

    void set_erode_size(int erode_px);
    void set_dilate_size(int dilate_px);
    
    void set_decorate (bool decorate){
        decorate = decorate;
    }

private:

    // Sizes of the erode and dilate blocks
    int erode_px, dilate_px;
    //cv::Size erode_size, dilate_size;
    bool erode_on, dilate_on;
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
    double mm_per_px; // TODO: impelment mm_per_px somehow

    std::string status_text;
    bool object_found;
    double object_area;
    cv::Point2i xy_coord_px;
    bool decorate; 

    unsigned int max_num_contours;
    double min_object_area;
    double max_object_area;
    
    // Mat client object for receiving frames
    MatClient frame_source;
    
    // Position server
    SMServer<shmem::Position2D, cv::Point2i> position_sink;
    
    // Mat server for sending processed frames
    bool frame_sink_used;
    MatServer frame_sink;

    // HSV filter
    void hsvTransform(cv::Mat& rgb_img);

    // Binary threshold
    void applyThreshold(const cv::Mat hsv_img, cv::Mat& threshold_img);
    
    // Use the binary threshold to mask the image
    void applyThresholdMask(const cv::Mat threshold_img, cv::Mat& hsv_img);

    // Erode/dilate objects to get rid of speckles
    void clarifyObjects(cv::Mat& threshold_img);

    // Add information to displayed image
    void decorateFeed(cv::Mat& display_img, const cv::Scalar&); // TODO: Remove (this class is doing too much!)
    
    // Callbacks for sliders
    static void erodeSliderChangedCallback(int, void*);
    static void dilateSliderChangedCallback(int, void*);
};

#endif	/* HSVFILTER_H */

