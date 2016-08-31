//******************************************************************************
//* File:   DifferenceDetector.h
//* Author: Jon Newman <jpnewman snail mit dot edu>
//
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu)
//* All right reserved.
//* This file is part of the Oat project.
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

#ifndef OAT_DIFFERENCEDETECTOR_H
#define	OAT_DIFFERENCEDETECTOR_H

#include "PositionDetector.h"

#include <limits>

namespace oat {

// Forward decl.
class Position2D;

/**
 * Motion-based object position detector.
 */
class DifferenceDetector : public PositionDetector {
public:

    /**
     * Motion-based object position detector.
     * @param frame_source_address Frame SOURCE node address
     * @param position_sink_address Position SINK node address
     */
    DifferenceDetector(const std::string &frame_source_address,
                       const std::string &position_sink_address);

    /**
     * Perform motion-based object position detection.
     * @param frame frame to look for object in.
     * @return  detected object position.
     */
    void detectPosition(cv::Mat &frame, oat::Position2D &position) override;

    void appendOptions(po::options_description &opts) override;
    void configure(const po::variables_map &vm) override;

    //Accessors (used for tuning GUI)
    void set_min_object_area(double value) { min_object_area_ = value; }
    void set_max_object_area(double value) { max_object_area_ = value; }
    void set_blur_size(int value);

private:

    // Intermediate variables
    cv::Mat this_image_, last_image_;
    cv::Mat threshold_frame_;
    bool last_image_set_ {false};

    // Object detection
    double object_area_ {0.0};

    // Detector parameters
    int difference_intensity_threshold_ {10};
    cv::Size blur_size_;
    bool blur_on_ {false};
    double min_object_area_ {0.0};
    double max_object_area_ {std::numeric_limits<double>::max()};

    // Tuning stuff
    const std::string tuning_image_title_;
    cv::Mat tune_frame_;
    int dummy0_ {0}, dummy1_ {10000};

    // Processing functions
    bool tuning_on_ {false};
    bool tuning_windows_created_ {false};
    void createTuningWindows(void);
    void tune(cv::Mat &frame, const oat::Position2D &position);
    void applyThreshold(cv::Mat &frame);

};

// Tuning GUI callbacks
void diffDetectorBlurSliderChangedCallback(int value, void *);
void diffDetectorMinAreaSliderChangedCallback(int value, void *);
void diffDetectorMaxAreaSliderChangedCallback(int value, void *);

}       /* namespace oat */
#endif	/* OAT_DIFFERENCEDETECTOR_H */

