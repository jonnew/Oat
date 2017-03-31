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
#include "Tuner.h"

#include <limits>

namespace oat {

class SimpleThreshold : public PositionDetector {

public:
    /**
     * Intensity threshold based object position detector for mono frame
     * streams.
     * @param frame_source_address Frame SOURCE node address
     * @param pose_sink_address Position SINK node address
     */
    SimpleThreshold(const std::string &frame_source_address,
                    const std::string &pose_sink_address);

private:
    // Configurable Interface
    po::options_description options() const override;
    void applyConfiguration(const po::variables_map &vm,
                            const config::OptionTable &config_table) override;

    // PositionDetector Interface
    bool checkPixelColor(oat::Pixel::Color c) override;
    oat::Pose detectPose(oat::Frame &frame) override;

    // Intermediate variables
    cv::Mat threshold_frame_;

    // Object detection
    double object_area_{0.0};

    // Internal matricies
    cv::Mat erode_element_, dilate_element_;

    // Detector parameters
    int t_min_{0};
    int t_max_{256};
    double min_object_area_{0.0};
    double max_object_area_{std::numeric_limits<double>::max()};

    // Erode and dilate kernels
    int erode_px_{0}, dilate_px_{0};
    bool makeEroder(int erode_px);
    bool makeDilater(int dilate_px);

    // Tuner
    std::unique_ptr<Tuner> tuner_{nullptr};
};

}       /* namespace oat */
#endif	/* OAT_SIMPLETHRESHOLD_H */
