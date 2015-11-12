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

namespace oat {

// Forward decl.
class Position2D;

/**
 * Motion-based object position detector.
 */
class DifferenceDetector2D : public PositionDetector {
public:

    /**
     * Motion-based object position detector.
     * @param frame_source_address Frame SOURCE node address
     * @param position_sink_address Position SINK node address
     */
    DifferenceDetector2D(const std::string &frame_source_address,
                         const std::string &position_sink_address);

    /**
     * Perform motion-based object position detection.
     * @param frame frame to look for object in.
     * @return  detected object position.
     */
    Position2D detectPosition(cv::Mat &frame) override;

    void configure(const std::string &config_file, const std::string &key);
private:

    // Intermediate variables
    cv::Mat this_image_, last_image_;
    cv::Mat threshold_image_;
    bool last_image_set_;

    // Object detection
    double object_area_;

    // The detected object position
    Position2D object_position_;

    // Detector parameters
    int difference_intensity_threshold_;
    cv::Size blur_size_;
    bool blur_on_;

    // Tuning stuff
    bool tuning_windows_created_;
    const std::string tuning_image_title_;
    cv::Mat tune_image_;
    void tune(void);
    void createTuningWindows(void);
    static void blurSliderChangedCallback(int value, void *);

    // Processing segregation
    // TODO: These are terrible - no IO signature other than void -> void,
    // no exceptions, etc
    void applyThreshold(void);
    void set_blur_size(int value);
    void siftBlobs(void);
    void servePosition(void);
};

}       /* namespace oat */
#endif	/* OAT_DIFFERENCEDETECTOR_H */

