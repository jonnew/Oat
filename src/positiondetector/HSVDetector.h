//******************************************************************************
//* File:   HSVDetector.h
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

#ifndef OAT_HSVDETECTOR_H
#define	OAT_HSVDETECTOR_H

#include "PositionDetector.h"
#include "Tuner.h"

#include <limits>
#include <string>

#include <opencv2/core/mat.hpp>

namespace oat {

class HSVDetector : public PositionDetector {

public:
    /**
     * A color-based object position detector with default parameters.
     * @param frame_source_address Frame SOURCE node address
     * @param position_sink_address Position SINK node address
     */
    HSVDetector(const std::string &frame_source_address,
                const std::string &position_sink_address);

private:
    // Configurable Interface
    po::options_description options() const override;
    void applyConfiguration(const po::variables_map &vm,
                            const config::OptionTable &config_table) override;

    //PositionDetector Interface
    oat::Pose detectPose(oat::Frame &frame) override;

    // Erode and dilate kernels
    int erode_px_{0}, dilate_px_{0};
    bool makeEroder(int erode_px);
    bool makeDilater(int dilate_px);

    // Internal matricies
    cv::Mat erode_element_, dilate_element_;

    // HSV threshold values
    int h_min_ {0}, h_max_ {256};
    int s_min_ {0}, s_max_ {256};
    int v_min_ {0}, v_max_ {256};

    // Detect object area
    double object_area_ {0.0};
    double min_object_area_ {0.0};
    double max_object_area_ {std::numeric_limits<double>::max()};

    // Tuner
    std::unique_ptr<Tuner> tuner_ {nullptr};
};

}       /* namespace oat */
#endif	/* OAT_HSVDETECTOR_H */
