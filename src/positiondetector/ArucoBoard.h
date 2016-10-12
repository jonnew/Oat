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

#include <string>
#include <opencv2/core/mat.hpp>
#include <opencv2/aruco.hpp>

namespace oat {

// Forward decl.
class Position2D;

/**
 * Aruco marker board marker tracking.
 */
class ArucoBoard : public PositionDetector {
public:

    /**
     * ArucoBoard marker tracking.
     * @param frame_source_address Frame SOURCE node address
     * @param position_sink_address Position SINK node address
     */
    ArucoBoard(const std::string &frame_source_address,
               const std::string &position_sink_address);

    /**
     * Perform aruco marker code detection
     * @param frame frame to look for object in.
     * @return detected marker position (upper left hand corner)
     */
    void detectPosition(cv::Mat &frame, oat::Position2D &position) override;

    void appendOptions(po::options_description &opts) override;
    void configure(const po::variables_map &vm) override;

private:

    // TODO: User defined dict
    /// Marker ID to look for
    cv::Ptr<cv::aruco::Board> board_;
    cv::Ptr<cv::aruco::DetectorParameters> detection_params_;

    cv::Matx33d camera_matrix_  {cv::Matx33d::eye()};
    std::vector<double> dist_coeff_ {0, 0, 0, 0, 0, 0, 0, 0};

    // TODO: program option
    cv::aruco::PREDEFINED_DICTIONARY_NAME
        marker_dict_id_ {cv::aruco::DICT_4X4_50};

    /// Board origin
    cv::Point3f origin_ {0, 0, 0};

    /// Board corner
    cv::Point3f corner_ {0, 0, 0};

    // TODO: For plotting, remove
    std::vector<cv::Point3f> ref_pts_;

    // TODO: Add options for dectection parameters

    // TODO: helper mode to print board. Hmm, or just point to external resource...
    //void printBoard(int board_id);
};

// Free standing functions
cv::Ptr<cv::aruco::Dictionary> getBoardDictionary(const std::string &key);

}       /* namespace oat */
#endif	/* OAT_ARUCOBOARD_H */
