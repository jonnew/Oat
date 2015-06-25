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

#include "PositionDetector.h"

/**
 * A color-based object position detector
 */
class HSVDetector2D : public Detector2D {
public:

    /**
     * A color-based object position detector with default parameters.
     * @param source_name Image SOURCE name
     * @param pos_sink_name Position SINK name
     */
    HSVDetector2D(const std::string& source_name, const std::string& pos_sink_name);


    /**
     * Perform color-based object position detection.
     * @param frame frame to look for object in.
     * @return  detected object position.
     */
    oat::Position2D detectPosition(cv::Mat& frame);
    
    void configure(const std::string& config_file, const std::string& config_key);

    // Accessors
    void set_erode_size(int erode_px);
    void set_dilate_size(int dilate_px);
    
private:
    
    // Sizes of the erode and dilate blocks
    int erode_px, dilate_px;
    bool erode_on, dilate_on;
    cv::Mat hsv_image, threshold_image, erode_element, dilate_element;

    // HSV threshold values
    int h_min, h_max;
    int s_min, s_max;
    int v_min, v_max;

    // Detect object area 
    double object_area;
    int min_object_area;
    int max_object_area;
    static constexpr double PI{3.14159265358979323846};
    
    // The detected object position
    oat::Position2D object_position;

    // Processing segregation 
    // TODO: These are terrible - no IO signature other than void -> void,
    
    // Binary threshold and use the binary threshold to mask the image
    void applyThreshold(void);

    // Erode/dilate objects to get rid of speckles
    void clarifyBlobs(void);
    
    // Sift through thresholded blobs to pull out potential object
    void siftBlobs(void);
   
    // Parameter tuning GUI functions and properties
    bool tuning_on;
    bool tuning_windows_created;
    const std::string tuning_image_title;
    cv::Mat tune_image;
    virtual void tune(void);
    virtual void createTuningWindows(void);
    static void erodeSliderChangedCallback(int, void*);
    static void dilateSliderChangedCallback(int, void*);
    

};

#endif	/* HSVFILTER_H */

