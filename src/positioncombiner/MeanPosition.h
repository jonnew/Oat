//******************************************************************************
//* File:   MeanPosition.h
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

#ifndef OAT_MEANPOSITION_H
#define	OAT_MEANPOSITION_H

#include "PositionCombiner.h"

#include <string>
#include <vector>

namespace oat {

/**
 * A mean position combiner.
 * A mean position combiner to generate a mean position from 2 or more
 * source positions. Can be used to generate the mean position heading
 * using a specified anchor position as a reference.
 */
class MeanPosition : public PositionCombiner {

    // Configurable Interface
    po::options_description options() const override;
    void applyConfiguration(const po::variables_map &vm,
                            const config::OptionTable &config_table) override;

    /**
     * Calculate the mean of SOURCE positions.
     * @param sources SOURCE positions to combine
     * @param combined_position Combined position output
     */
    void combine(const std::vector<oat::Position2D> &source_positions,
                 oat::Position2D &combined_position) override;

    /// Should a heading be generated?
    bool generate_heading_ {false};

    /// SOURCE position to be used ad an anchor when calculating heading
    int heading_anchor_idx_ {0};
};

}      /* namespace oat */
#endif /* OAT_MEANPOSITION_H */

