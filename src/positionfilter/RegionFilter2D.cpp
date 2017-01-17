//******************************************************************************
//* File:   RegionFilter.cpp
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

#include <ostream>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <string.h>
#include <vector>
#include <cpptoml.h>

#include "RegionFilter2D.h"

#include "../../lib/utility/TOMLSanitize.h"
#include "../../lib/utility/IOFormat.h"

namespace oat {

RegionFilter2D::~RegionFilter2D()
{
    for (auto &value : region_contours_) {
        delete value;
    }
}

po::options_description RegionFilter2D::options() const
{
    // Update CLI options
    po::options_description local_opts;
    local_opts.add_options()
        ("regions", po::value<std::string>(),
         "NOTE: Regions can only be specified in a config file.\n"
         "Regions contours are specified as n-point matrices, [[x0, y0],[x1, "
         "y1],...,[xn, yn]], which define the vertices of a polygon:\n\n"
         "  <region> = [[+float, +float],\n"
         "              [+float, +float],\n"
         "              ...              \n"
         "              [+float, +float]]\n\n"
         "The name of the contour is used as the region label (10 characters "
         "max). For example, here is an octagonal region called CN and a "
         "tetragonal region called R0:\n\n"
         "  CN = [[336.00, 272.50],\n"
         "        [290.00, 310.00],\n"
         "        [289.00, 369.50],\n"
         "        [332.67, 417.33],\n"
         "        [389.33, 413.33],\n"
         "        [430.00, 375.33],\n"
         "        [433.33, 319.33],\n"
         "        [395.00, 272.00]]\n\n"
         "  R0 = [[654.00, 380.00],\n"
         "        [717.33, 386.67],\n"
         "        [714.00, 316.67],\n"
         "        [655.33, 319.33]]")
        ;

    return local_opts;
}

void RegionFilter2D::applyConfiguration(const po::variables_map &vm,
                                        const config::OptionTable &config_table)
{
    // The config should be an table of arrays.
    // Each key specifies the region ID and its value specifies an array
    // defining a vector of 2D points.
    if (vm.count("regions"))
        throw std::runtime_error("Regions can only be specified using a config file.");

    // Iterate through each region definition
    auto it = config_table->begin();

    while (it != config_table->end()) {

        oat::config::Array region_array;
        oat::config::getArray(config_table, it->first, region_array);

        // Push the name of this region onto the id list
        region_ids_.push_back(it->first);
        if (region_ids_.back().size() > oat::Position2D::REGION_LEN)
            std::cerr << oat::Warn("Region names are limited to 10 characters.");

        region_contours_.push_back(new std::vector<cv::Point>());

        auto region = region_array->nested_array();
        auto reg_it = region.begin();

        while (reg_it != region.end()) {

            auto point = (**reg_it).array_of<double>();

            if (point.size() != 2) {
                throw std::runtime_error("Each region must be a nested, Nx2"
                                         "TOML array of doubles to specify a "
                                         "polygon contour");
            }

            auto p = cv::Point2d(point[0]->get(), point[1]->get());
            region_contours_.back()->push_back(p);
            reg_it++;
        }
        it++;
    }
}

void RegionFilter2D::filter(oat::Position2D &position) {

    // Check the current position to see if it lies inside any regions.
    if (position.position_valid) {

        cv::Point pt = (cv::Point)position.position;
        std::vector<std::string>::size_type i = 0;

        for (auto &r : region_contours_) {

            if (cv::pointPolygonTest(*r, pt, false) >= 0) {

                position.region_valid = true;
                std::vector<char> writable(region_ids_[i].begin(),
                                           region_ids_[i].end());
                writable.push_back('\0');
                strcpy(position.region, &writable[0]);
                break;
            }
            i++;
        }
    }
}

} /* namespace oat */
