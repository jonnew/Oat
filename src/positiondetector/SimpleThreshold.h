//******************************************************************************
//* File:   SimpleThreshold.h
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

#ifndef OAT_SIMPLETHRESHOLD_H
#define	OAT_SIMPLETHRESHOLD_H

#include "PositionDetector.h"

#include <limits>

namespace oat {

// Forward decl.
class Position2D;

/**
 * Motion-based object position detector.
 */
class SimpleThreshold : public PositionDetector {
public:

    /**
     * Intensity threshold based object position detector for mono frame
     * streams.
     * @param frame_source_address Frame SOURCE node address
     * @param position_sink_address Position SINK node address
     */
    SimpleThreshold(const std::string &frame_source_address,
                    const std::string &position_sink_address);

    void detectPosition(cv::Mat &frame, oat::Position2D &position) override;

    void appendOptions(po::options_description &opts) override;
    void configure(const po::variables_map &vm) override;

    //Accessors (used for tuning GUI)
    void set_min_object_area(double value) { min_object_area_ = value; }
    void set_max_object_area(double value) { max_object_area_ = value; }
    void set_erode_size(int erode_px);
    void set_dilate_size(int dilate_px);
    void set_mincomp_size(int mincomp_val);

private:

    // Intermediate variables
    cv::Mat threshold_frame_;
    cv::Mat nonmasked_frame_;

    // Object detection
    double object_area_ {0.0};

    // Sizes of the erode and dilate blocks
    int erode_px_ {0}, dilate_px_ {0}, mincomp_val_ {0};
    bool erode_on_ {false}, dilate_on_ {false}, mincomp_on_ {false};

    // Internal matricies
    cv::Mat erode_element_, dilate_element_;

    // Detector parameters
    int t_min_ {0};
    int t_max_ {256};
    double min_object_area_ {0.0};
    double max_object_area_ {std::numeric_limits<double>::max()};

    // Tuning stuff
    bool tuning_on_ {false};
    bool tuning_windows_created_ {false};
    const std::string tuning_image_title_;
    cv::Mat tune_frame_;
    int dummy0_ {0}, dummy1_ {100000};

    // Processing functions
    void createTuningWindows(void);
    void tune(cv::Mat &frame, const oat::Position2D &position);
    void applyThreshold(cv::Mat &frame);
};

// Tuning GUI callbacks
void simpleThresholdMinAreaSliderChangedCallback(int value, void *);
void simpleThresholdMaxAreaSliderChangedCallback(int value, void *);
void simpleThresholdErodeSliderChangedCallback(int value, void *);
void simpleThresholdDilateSliderChangedCallback(int value, void *);

}       /* namespace oat */
#endif	/* OAT_SIMPLETHRESHOLD_H */
