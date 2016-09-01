//******************************************************************************
//* File:   HomgraphicTransform2D.cpp
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
//******************************************************************************

#include <string>
#include <cpptoml.h>

#include "../../lib/utility/TOMLSanitize.h"
#include "../../lib/utility/IOFormat.h"

#include "HomographyTransform2D.h"

namespace oat {

HomographyTransform2D::HomographyTransform2D(const std::string &position_source_address,
                                             const std::string& position_sink_address) :
PositionFilter(position_source_address, position_sink_address)
{
    // Nothing
}

void HomographyTransform2D::appendOptions(po::options_description &opts) {

    // Accepts a config file
    PositionFilter::appendOptions(opts);

    // Update CLI options
    po::options_description local_opts;
    local_opts.add_options()
        ("homography,H", po::value<std::string>(),
         "A nine-element array of floats, [h11,h12,...,h33], specifying a "
         "homography matrix for 2D position. Generally produced by "
         "oat-calibrate homography.")
        ;

    opts.add(local_opts);

    // Return valid keys
    for (auto &o: local_opts.options())
        config_keys_.push_back(o->long_name());
}

void HomographyTransform2D::configure(const po::variables_map &vm) {

    // Check for config file and entry correctness
    auto config_table = oat::config::getConfigTable(vm);
    oat::config::checkKeys(config_keys_, config_table);
    
    // Homography
    std::vector<double> H;
    if (oat::config::getArray<double, 9>(vm, config_table, "homography", H)) {

        homography_(0, 0) = H[0];
        homography_(0, 1) = H[1];
        homography_(0, 2) = H[2];
        homography_(1, 0) = H[3];
        homography_(1, 1) = H[4];
        homography_(1, 2) = H[5];
        homography_(2, 0) = H[6];
        homography_(2, 1) = H[7];
        homography_(2, 2) = H[8];
    }
}

void HomographyTransform2D::filter(oat::Position2D& position) {

    // TODO: If the homography_is not valid, I should warn the user...
    if (homography_valid_) {

        // Position transform
        if (position.position_valid) {
            std::vector<oat::Point2D> in_positions;
            std::vector<oat::Point2D> out_positions;
            in_positions.push_back(position.position);
            cv::perspectiveTransform(in_positions, out_positions, homography_);
            position.position = out_positions[0];
        }

        // Velocity transform
        if (position.velocity_valid) {
            std::vector<oat::Velocity2D> in_velocities;
            std::vector<oat::Velocity2D> out_velocities;
            cv::Matx33d vel_homo = homography_;
            vel_homo(0, 2) = 0.0; // offsets do not apply to velocity
            vel_homo(1, 2) = 0.0; // offsets do not apply to velocity
            in_velocities.push_back(position.velocity);
            cv::perspectiveTransform(in_velocities, out_velocities, vel_homo);
            position.velocity = out_velocities[0];
        }

        // Heading transform
        if (position.heading_valid) {
            std::vector<oat::UnitVector2D> in_heading;
            std::vector<oat::UnitVector2D> out_heading;
            cv::Matx33d head_homo = homography_;
            head_homo(0, 2) = 0.0; // offsets do not apply to heading
            head_homo(1, 2) = 0.0; // offsets do not apply to heading
            in_heading.push_back(position.heading);
            cv::perspectiveTransform(in_heading, out_heading, head_homo);
            cv::normalize(out_heading, out_heading);
            position.heading = out_heading[0];
        }

        // Update outgoing position's coordinate system
        position.setCoordSystem(oat::DistanceUnit::WORLD, homography_);
    }
}


} /* namespace oat */
