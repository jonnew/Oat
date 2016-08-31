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

#include <string>
#include <vector>
#include <cmath>
#include <cpptoml.h>

#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/TOMLSanitize.h"

#include "MeanPosition.h"

namespace oat {

void MeanPosition::appendOptions(po::options_description &opts) {

    // Accepts a config file and sources/sinks
    // TODO: Code smell -- this is required in order for specializations to
    // function at runtime...
    PositionCombiner::appendOptions(opts);

    // Update CLI options
    po::options_description local_opts;
    local_opts.add_options()
        ("heading-anchor,h", po::value<int>(),
         "Index of the SOURCE position to use as an anchor when calculating "
         "object heading. In this case the heading equals the mean directional "
         "vector between this anchor position and all other SOURCE positions. If "
         "unspecified, the heading is not calculated.")
        ;

    opts.add(local_opts);

    // Populate valid keys
    for (auto &o: local_opts.options())
        config_keys_.push_back(o->long_name());
}

void MeanPosition::configure(const po::variables_map &vm) {

    // Setup sources and sink
    // TODO: Code smell -- this is required in order for specializations to
    // function at runtime...
    PositionCombiner::configure(vm);

    // Check for config file and entry correctness
    auto config_table = oat::config::getConfigTable(vm);
    //oat::config::checkKeys(config_keys_, config_table);

    // Adaptation coefficient
    generate_heading_ = oat::config::getNumericValue<int>(
        vm, config_table, "heading-anchor", heading_anchor_idx_, 0, num_sources() - 1
    );
}

void MeanPosition::combine(const std::vector<oat::Position2D> &sources,
                           oat::Position2D &combined_position) {

    double mean_denom = 1.0/(double) sources.size();
    combined_position.position = oat::Point2D(0,0);
    combined_position.position_valid = true;
    combined_position.velocity = oat::Velocity2D(0,0);
    combined_position.velocity_valid = true;
    combined_position.heading = oat::UnitVector2D(0,0);
    combined_position.heading_valid = true;

    // Averaging operation
    for (auto &pos : sources) {

        // Position
        if (pos.position_valid)
            combined_position.position += mean_denom * pos.position;
        else
            combined_position.position_valid = false;

        // Velocity
        if (pos.velocity_valid)
            combined_position.velocity += mean_denom * pos.velocity;
        else
            combined_position.velocity_valid = false;

        if (generate_heading_) {
            // Find heading from anchor to other positions
            if (combined_position.position_valid) {

                oat::Point2D diff =  pos.position - sources[heading_anchor_idx_].position;
                combined_position.heading += diff;

            } else {
                combined_position.heading_valid = false;
            }

        } else {

            // Head direction
            if (pos.heading_valid)
                combined_position.heading += pos.heading;
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
}

} /* namespace oat */
