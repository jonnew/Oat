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

#include "Detector2D.h"
#include "../../lib/datatypes/Position2D.h"

#define PI 3.14159265358979323846

class HSVDetector2D : public Detector2D {
    
public:
    
    // HSV values known
    HSVDetector2D(const std::string& source_name, const std::string& pos_sink_name,
                int h_min, int h_max,
                int s_min, int s_max,
                int v_min, int v_max);
    
    // Start with full range of HSV thresholds. This is typically used
    // for manual tuning of thresholds using the createTrackbars call
    HSVDetector2D(const std::string& source_name, const std::string& pos_sink_name);

    // Use a configuration file to specify parameters
    void configure(const std::string& config_file, const std::string& config_key);
    
    // Apply the HSVTransform, thresholding, and erode/dilate operations to/from
    // shared memory allocated mat objects
    oat::Position2D detectPosition(cv::Mat& frame_in);

    // Accessors
    std::string get_detector_name() { return name; }
    void set_min_object_area(double value) { min_object_area = value; }
    void set_max_object_area(double value) { max_object_area = value; }
    void set_erode_size(int erode_px);
    void set_dilate_size(int dilate_px);
    
private:
    
    // Sizes of the erode and dilate blocks
    int erode_px, dilate_px;
    bool erode_on, dilate_on;
    cv::Mat hsv_image, threshold_image, erode_element, dilate_element;

    // HSV threshold values
    int h_min;
    int h_max;
    int s_min;
    int s_max;
    int v_min;
    int v_max;

    // For manual manipulation of HSV filtering
    const std::string name;

    // Object detection
    double object_area;
    
    // The detected object position
    oat::Position2D object_position;

    double min_object_area;
    double max_object_area;
    
    // Processing segregation 
    // TODO: These are terrible - no IO sigature other than void -> void,
    // no exceptions, etc
    
    // Binary threshold and use the binary threshold to mask the image
    void applyThreshold(void);

    // Erode/dilate objects to get rid of speckles
    void clarifyBlobs(void);
    
    // Sift through thresholded blobs to pull out potential object
    void siftBlobs(void);
    
    // Tuning stuff
    bool tuning_windows_created;
    const std::string tuning_image_title;
    cv::Mat tune_image;
    virtual void tune(void);
    virtual void createTuningWindows(void);
    static void erodeSliderChangedCallback(int, void*);
    static void dilateSliderChangedCallback(int, void*);
};

#endif	/* HSVFILTER_H */

