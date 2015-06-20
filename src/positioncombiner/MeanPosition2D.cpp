//******************************************************************************
//* Copyright (c) Jon Newman (jpnewman at mit snail edu) 
//* All right reserved.
//* This file is part of the Simple Tracker project.
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
#include "MeanPosition2D.h"

MeanPosition2D::MeanPosition2D(std::vector<std::string> position_source_names, std::string sink_name) :
  PositionCombiner(position_source_names, sink_name)
, generate_heading(false)
, heading_anchor_idx(0) { }

void MeanPosition2D::configure(const std::string& config_file, const std::string& config_key) {
    
    cpptoml::table config;

    try {
        config = cpptoml::parse_file(config_file);
    } catch (const cpptoml::parse_exception& e) {
        std::cerr << "Failed to parse " << config_file << ": " << e.what() << std::endl;
    }

    try {
        // See if a camera configuration was provided
        if (config.contains(config_key)) {

            auto this_config = *config.get_table(config_key);

            if (this_config.contains("generate-heading")) {
                generate_heading = *this_config.get_as<bool>("generate-heading");
            }
            
            if (this_config.contains("heading-anchor")) {
                
                if (!generate_heading) {
                    std::cerr << "Position combiner is not set to generate heading, so specifying a heading anchor does not make sense.\n"
                              << "heading-anchor option ignored.\n";
                } else {
                    heading_anchor_idx = *this_config.get_as<int64_t>("heading-anchor");
                }
                
                if (heading_anchor_idx >= get_number_of_sources() || heading_anchor_idx < 0) {
                    std::cerr << "Specified heading-anchor exceeds the number of position sources or is < 0.\n"
                              << "heading-anchor set to 0.\n";
                    heading_anchor_idx = 0;
                }
            }

        } else {
            std::cerr << "No position combiner configuration named \"" + config_key + "\" was provided. Exiting." << std::endl;
            exit(EXIT_FAILURE);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}

/**
 * This function calculates the geometric mean of all source positions.
 * @param sources Source positions
 * @return Combined position
 */
oat::Position2D MeanPosition2D::combinePositions(const std::vector<oat::Position2D*>& sources) {

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