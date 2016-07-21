//******************************************************************************
//* File:   Aruco.h
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

#ifndef OAT_ARUCO_H
#define	OAT_ARUCO_H

#include <string>
#include <opencv2/core/mat.hpp>

#include "PositionDetector.h"

namespace oat {

// Forward decl.
class Position2D;

enum class HeadingDirection {
    NW = 0,
    NE,
    SE,
    SW
};

/**
 * Aruco marker tracking.
 */
class Aruco : public PositionDetector {
public:

    /**
     * Aruco marker tracking.
     * @param frame_source_address Frame SOURCE node address
     * @param position_sink_address Position SINK node address
     */
    Aruco(const std::string &frame_source_address,
          const std::string &position_sink_address);

    /**
     * Perform aruco marker code detection
     * @param frame frame to look for object in.
     * @return detected marker position (upper left hand corner)
     */
    void detectPosition(cv::Mat &frame, oat::Position2D &position) override;

    void configure(const std::string &config_file,
                   const std::string &config_key) override;

private:

    /// Marker ID to look for
    int marker_id_ {1};

    /// TODO: Marker dictionary to use

    /// TODO: Heading direction
    oat::HeadingDirection heading_dir_ {oat::HeadingDirection::NW};

    /// TODO: Does this belong here?
    // TODO: cv::Matx33<double> 
    //bool perspective_defined_ {false};
    //cv::Mat distortion_coeffs;
    //cv::Mat perspective_transform_;
};

}       /* namespace oat */
#endif	/* OAT_ARUCO_H */
