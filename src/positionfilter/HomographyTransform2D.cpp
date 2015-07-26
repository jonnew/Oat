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

#include "HomographyTransform2D.h"

#include "../../lib/cpptoml/cpptoml.h"
#include "../../lib/cpptoml/OatTOMLSanitize.h"
#include "../../lib/utility/IOFormat.h"

HomographyTransform2D::HomographyTransform2D(const std::string& position_source_name, const std::string& position_sink_name) :
PositionFilter(position_source_name, position_sink_name)
, homography_valid(false)
, homography(1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0) { }

oat::Position2D HomographyTransform2D::filterPosition(oat::Position2D& raw_position) {
    
    filtered_position = raw_position;

    // Position transforms
    std::vector<oat::Point2D> in_positions;
    std::vector<oat::Point2D> out_positions;
    in_positions.push_back(raw_position.position);
    cv::perspectiveTransform(in_positions, out_positions, homography);
    filtered_position.position = out_positions[0];

    // Velocity transform
    std::vector<oat::Velocity2D> in_velocities;
    std::vector<oat::Velocity2D> out_velocities;
    cv::Matx33d vel_homo = homography;
    vel_homo(0, 2) = 0.0; // offsets do not apply to velocity
    vel_homo(1, 2) = 0.0; // offsets do not apply to velocity
    in_velocities.push_back(raw_position.velocity);
    cv::perspectiveTransform(in_velocities, out_velocities, vel_homo);
    filtered_position.velocity = out_velocities[0];

    // Head direction is normalized and unit-free, and therefore
    // does not require conversion
    // TODO: No, this is not true. what about shear distortion??
    
    // Return value uses world coordinates
    if (homography_valid)
        filtered_position.coord_system = oat::WORLD;
    
    return filtered_position;
}

void HomographyTransform2D::configure(const std::string& config_file, const std::string& config_key) {

    // Available options
    std::vector<std::string> options {"homography"};
    
    // This will throw cpptoml::parse_exception if a file 
    // with invalid TOML is provided
    cpptoml::table config;
    config = cpptoml::parse_file(config_file);

    // See if a camera configuration was provided
    if (config.contains(config_key)) {

        // Get this components configuration table
        auto this_config = config.get_table(config_key);

        // Check for unknown options in the table and throw if you find them
        oat::config::checkKeys(options, this_config);
        
        // Homography matrix
        oat::config::Array homo_array;
        if (oat::config::getArray(this_config, "homography", homo_array, 9, true)) {

            auto homo_vec = homo_array->array_of<double>();
            
            homography(0, 0) = homo_vec[0]->get();
            homography(0, 1) = homo_vec[1]->get();
            homography(0, 2) = homo_vec[2]->get();
            homography(1, 0) = homo_vec[3]->get();
            homography(1, 1) = homo_vec[4]->get();
            homography(1, 2) = homo_vec[5]->get();
            homography(2, 0) = homo_vec[6]->get();
            homography(2, 1) = homo_vec[7]->get();
            homography(2, 2) = homo_vec[8]->get();

            homography_valid = true;
        }
    } else {
        throw (std::runtime_error(oat::configNoTableError(config_key, config_file)));
    }
}
