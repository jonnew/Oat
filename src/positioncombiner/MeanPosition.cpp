//******************************************************************************
//* File:   MeanPosition.cpp
//* Author: Jon Newman <jpnewman snail mit dot edu>
//*
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

#include <vector>
#include <cmath>

#include "../../lib/cpptoml/cpptoml.h"
#include "../../lib/cpptoml/OatTOMLSanitize.h"

#include "MeanPosition.h"

MeanPosition::MeanPosition(std::vector<std::string> position_source_names, std::string sink_name) :
  PositionCombiner(position_source_names, sink_name)
, generate_heading(false)
, heading_anchor_idx(0) { }

void MeanPosition::configure(const std::string& config_file, const std::string& config_key) {
    
    // Available options
    std::vector<std::string> options {"heading_anchor"};
    
    // This will throw cpptoml::parse_exception if a file 
    // with invalid TOML is provided
    cpptoml::table config;
    config = cpptoml::parse_file(config_file);

    // See if a configuration was provided
    if (config.contains(config_key)) {

        // Get this components configuration table
        auto this_config = config.get_table(config_key);
        
        // Check for unknown options in the table and throw if you find them
        oat::config::checkKeys(options, this_config);

        // Heading anchor
        if (oat::config::getValue(this_config, "heading_anchor",
                heading_anchor_idx, 
                static_cast<int64_t>(0), 
                static_cast<int64_t>(get_number_of_sources() - 1))) {

            generate_heading = true;
        }
        
    } else {
        throw (std::runtime_error(oat::configNoTableError(config_key, config_file)));
    } 
}

/**
 * This function calculates the geometric mean of all source positions.
 * @param sources Source positions
 * @return Combined position
 */
oat::Position2D MeanPosition::combinePositions(const std::vector<oat::Position2D*>& sources) {

    double mean_denom = 1.0/(double) sources.size(); 
    oat::Position2D combined_position;
    combined_position.position = oat::Point2D(0,0);
    combined_position.position_valid = true;
    combined_position.velocity = oat::Velocity2D(0,0);
    combined_position.velocity_valid = true;
    combined_position.heading = oat::UnitVector2D(0,0);
    combined_position.heading_valid = true;
    
    // Averaging operation
    for (auto pos : sources) {
        
        // Position
        if (pos->position_valid)
            combined_position.position += mean_denom * pos->position;
        else
            combined_position.position_valid = false;
        
        // Velocity
        if (pos->velocity_valid)
            combined_position.velocity += mean_denom * pos->velocity;
        else
            combined_position.velocity_valid = false;
        
        if (generate_heading) {
            // Find heading from anchor to other positions
            if (combined_position.position_valid) {
                
                oat::Point2D diff =  pos->position - sources[heading_anchor_idx]->position;
                combined_position.heading += diff;
                
            } else {
                combined_position.heading_valid = false; 
            }
            
        } else {

            // Head direction
            if (pos->heading_valid)
                combined_position.heading += pos->heading;
            else
                combined_position.heading_valid = false; 
        }
        
    }
    
    // Renormalize head-direction unit vector
    if (combined_position.heading_valid)
    {
        double mag = std::sqrt(
                std::pow(combined_position.heading.x, 2.0) +
                std::pow(combined_position.heading.y, 2.0));

        combined_position.heading = 
                combined_position.heading/mag;
    }
    
    return combined_position;
}