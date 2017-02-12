//******************************************************************************
//* File:   ArucoBoard.h
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

#ifndef OAT_ARUCOBOARD_H
#define	OAT_ARUCOBOARD_H

#include "PositionDetector.h"
#include "Tuner.h"

#include <string>
#include <opencv2/core/mat.hpp>
#include <opencv2/aruco.hpp>

namespace oat {

// Forward decl.
class Position2D;

class ArucoBoard : public PositionDetector {

    using Corners = std::vector<std::vector<cv::Point2f>>;
    using pGB = cv::Ptr<cv::aruco::GridBoard>;

public:
    /**
     * ArucoBoard marker tracking.
     */
    using PositionDetector::PositionDetector;

private:
    // Configurable Interface
    po::options_description options() const override;
    void applyConfiguration(const po::variables_map &vm,
                            const config::OptionTable &config_table) override;

    // PositionDetector Interface
    oat::Pose detectPose(oat::Frame &frame) override;

    // Intrinsic parameters
    cv::Matx33d camera_matrix_{cv::Matx33d::eye()};
    std::vector<double> dist_coeff_{0, 0, 0, 0, 0, 0, 0, 0};

    // Marker or marker board to look for
    float marker_length_;
    cv::Ptr<cv::aruco::Board> board_;

    // Marker detection parameters
    bool refine_detection_ {false};
    cv::Ptr<cv::aruco::DetectorParameters> dp_;

    // Tuner
    std::unique_ptr<Tuner> tuner_ {nullptr};
};

/**
 * @brief Get the ARUCO dictionary enum value from a string representation
 * @param key A string of the form <Size>X<Size>_<Number of Markers> specifying
 * the ARUCO dictionary to use.
 * @return id of the selected dictionary.
 */
int arucoDictionaryID(const std::string &key);

}       /* namespace oat */
#endif	/* OAT_ARUCOBOARD_H */
