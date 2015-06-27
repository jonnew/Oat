//******************************************************************************
//* File:   RegionFilter.cpp
//* Author: Jon Newman <jpnewman snail mit dot edu>
//
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu) 
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

#include <ostream>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <string.h>

#include "RegionFilter2D.h"

#include "../../lib/cpptoml/cpptoml.h"
#include "../../lib/utility/IOFormat.h"

RegionFilter2D::RegionFilter2D(const std::string& position_source_name, const std::string& position_sink_name) :
  PositionFilter(position_source_name, position_sink_name)
, regions_configured(false) { }

RegionFilter2D::~RegionFilter2D() {
    
    for (auto &value : region_contours) {
        delete value;
    }
}

void RegionFilter2D::configure(const std::string& config_file, const std::string& config_key) {
    
    // This will throw cpptoml::parse_exception if a file 
    // with invalid TOML is provided
    cpptoml::table config;
    config = cpptoml::parse_file(config_file);

    // See if a camera configuration was provided
    if (config.contains(config_key)) {

        // The config should be an array of tables, with one key-value
        // pair per table, with the key specifying the region ID and the 
        // Value an array specifying a vector of 2D points.

        auto region_tab = *config.get_table(config_key);

        auto it = region_tab.begin();

        while (it != region_tab.end()) {

            auto region_val = *it;

            if (!region_val.second->is_array()) {
                 throw std::runtime_error(
                         oat::configValueError(
                         region_val.first,
                         config_key, 
                         config_file,
                         "must be a nested, Nx2 TOML array of doubles to specify a region contour")
                         );
            }
            
            // Push the name of this region onto the id list
            region_ids.push_back(region_val.first);
            region_contours.push_back(new std::vector<cv::Point>());
            
            auto region = region_val.second->as_array()->nested_array();
            auto reg_it = region.begin();

            while (reg_it != region.end()) {

                // This should be a 2-element vector
                if (*reg_it == nullptr) {
                    throw std::runtime_error(
                         oat::configValueError(
                         region_val.first,
                         config_key, 
                         config_file,
                         "must be a nested, Nx2 TOML array of doubles to specify a region contour.")
                         );
                    
                }
                
                auto point = (**reg_it).array_of<double>();

                if (point.size() != 2) {
                    throw std::runtime_error(
                         oat::configValueError(
                         region_val.first,
                         config_key, 
                         config_file,
                         "must be a nested, Nx2 TOML array of doubles to specify a region contour")
                         );
                }

                auto p = cv::Point2d(point[0]->get(), point[1]->get());
                region_contours.back()->push_back(p);
                reg_it++;
            }
            it++;
        }

        //            //check the result
        //            for (int i = 0; i < region_contours.size(); i++) {
        //                std::cout << "Region ID: " + region_ids[i] + "\n";
        //                for (int j = 0; j < region_contours[i]->size(); j++) {
        //                    std::cout << "x: " + std::to_string(region_contours[i]->at(j).x) + " "
        //                              << "y: " + std::to_string(region_contours[i]->at(j).y) + "\n";
        //                }
        //            }

    } else {
         throw (std::runtime_error(oat::configNoTableError(config_key, config_file)));
    }
}

/**
 * Check the position to see if position->position lies within any of the 
 * contours defined in the configuration. In the case that the point lies within
 * multiple regions, the first one checked is used and the others are ignored.
 * @param position_in
 * @return 
 */
oat::Position2D RegionFilter2D::filterPosition(oat::Position2D& position_in) {
    
    // Check the current position to see if it lies inside any regions.
    if (position_in.position_valid) {

        cv::Point pt = (cv::Point)position_in.position;
        std::vector<std::string>::size_type i = 0;
        
        for (auto &r : region_contours) {
            
            if (cv::pointPolygonTest(*r, pt, false) >= 0) {
                
                position_in.region_valid = true;

                std::vector<char> writable(region_ids[i].begin(), region_ids[i].end());
                writable.push_back('\0');
                
                strcpy(position_in.region, &writable[0]);

                break;
            }
            
            i++;
        }
    }
    
    return position_in;
}

