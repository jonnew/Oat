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

#ifndef OAT_RPGPOSEEST_H
#define	OAT_RPGPOSEEST_H

#include "PositionDetector.h"
#include "Tuner.h"

#include <string>

#include "./rpgposeest/pose_estimator.h"

namespace oat {

namespace mpe = monocular_pose_estimator;

class RPGPoseEst : public PositionDetector {

    using Corners = std::vector<std::vector<cv::Point2f>>;
    using PE = mpe::PoseEstimator;

public:
    /**
     * @brief A Monocular Pose Estimation System based on LEDs
     * @note @inproceedings{Faessler2014ICRA,
     * author = {Faessler, Matthias and Mueggler, Elias and Schwabe, Karl and
     * Scaramuzza, Davide},
     * title = {A Monocular Pose Estimation System based on Infrared {LED}s},
     * booktitle = {IEEE International Conference on Robotics and Automation
     * (ICRA)},
     * year = {2014}
     * }
     */
    RPGPoseEst(const std::string &frame_source_address,
               const std::string &pose_sink_address);

private:
    // Configurable Interface
    po::options_description options() const override;
    void applyConfiguration(const po::variables_map &vm,
                            const config::OptionTable &config_table) override;

    // PositionDetector Interface
    void detectPosition(oat::Frame &frame, oat::Pose &pose) override;

    // The RPG monocoluar pose estimator
    PE tracker_;
};

}       /* namespace oat */
#endif	/* OAT_RPGPOSEEST_H */
