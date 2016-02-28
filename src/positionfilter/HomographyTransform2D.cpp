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

#include "../../lib/utility/OatTOMLSanitize.h"
#include "../../lib/utility/IOFormat.h"

#include "HomographyTransform2D.h"

namespace oat {

HomographyTransform2D::HomographyTransform2D(const std::string &position_source_address,
                                             const std::string& position_sink_address) :
PositionFilter(position_source_address, position_sink_address)
{
    // Nothing
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

void HomographyTransform2D::configure(const std::string &config_file,
                                      const std::string &config_key) {

    // Available options
    std::vector<std::string> options {"homography"};

    // This will throw cpptoml::parse_exception if a file
    // with invalid TOML is provided
    auto config = cpptoml::parse_file(config_file);

    // See if a camera configuration was provided
    if (config->contains(config_key)) {

        // Get this components configuration table
        auto this_config = config->get_table(config_key);

        // Check for unknown options in the table and throw if you find them
        oat::config::checkKeys(options, this_config);

        // Homography matrix
        oat::config::Array homo_array;
        if (oat::config::getArray(this_config, "homography", homo_array, 9, true)) {

            auto homo_vec = homo_array->array_of<double>();

            homography_(0, 0) = homo_vec[0]->get();
            homography_(0, 1) = homo_vec[1]->get();
            homography_(0, 2) = homo_vec[2]->get();
            homography_(1, 0) = homo_vec[3]->get();
            homography_(1, 1) = homo_vec[4]->get();
            homography_(1, 2) = homo_vec[5]->get();
            homography_(2, 0) = homo_vec[6]->get();
            homography_(2, 1) = homo_vec[7]->get();
            homography_(2, 2) = homo_vec[8]->get();

            homography_valid_ = true;
        }
    } else {
        throw (std::runtime_error(oat::configNoTableError(config_key, config_file)));
    }
}

} /* namespace oat */
