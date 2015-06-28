//******************************************************************************
//* File:   DifferenceDetector.h
//* Author: Jon Newman <jpnewman snail mit dot edu>
//
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu) 
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
//****************************************************************************

#ifndef DIFFERENCEDETECTOR_H
#define	DIFFERENCEDETECTOR_H

#include "PositionDetector.h"

/**
 * Motion-based object position detector.
 * @param image_source_name
 * @param position_sink_name
 */
class DifferenceDetector2D : public PositionDetector {
public:
    
    /**
     * Motion-based object position detector.
     * @param source_name Image SOURCE name
     * @param pos_sink_name Position SINK name
     */
    DifferenceDetector2D(const std::string& image_source_name, const std::string& pos_sink_name);

    /**
     * Perform motion-based object position detection.
     * @param frame frame to look for object in.
     * @return  detected object position.
     */
    oat::Position2D detectPosition(cv::Mat& frame);
    
    void configure(const std::string& config_file, const std::string& key);
private:
    
    // Intermediate variables
    cv::Mat this_image, last_image;
    cv::Mat threshold_image;
    bool last_image_set;
    
    // Object detection
    double object_area;
    
    // The detected object position
    oat::Position2D object_position;
    
    // Detector parameters
    int difference_intensity_threshold;
    cv::Size blur_size;
    bool blur_on;
    
    // Tuning stuff
    bool tuning_on;
    bool tuning_windows_created;
    const std::string tuning_image_title;
    cv::Mat tune_image;
    void tune(void);
    void createTuningWindows(void);
    static void blurSliderChangedCallback(int, void*);

    // Processing segregation 
    // TODO: These are terrible - no IO signature other than void -> void,
    // no exceptions, etc
    void applyThreshold(void);
    void set_blur_size(int value);
    void siftBlobs(void);
    void servePosition(void);

};

#endif	/* DIFFERENCEDETECTOR_H */

