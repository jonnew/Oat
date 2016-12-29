//******************************************************************************
//* File:   RegionFilter.h
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

#ifndef OAT_REGIONFILTER2D_H
#define	OAT_REGIONFILTER2D_H

#include "PositionFilter.h"

#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace oat {

// Forward decl.
class Position2D;

class RegionFilter2D : public PositionFilter {

public:
    /**
     * A region filter.
     * A region filter to map position coordinates to categorical regions. By
     * specifying a set of named contours, this filter checks if the position
     * is inside a given contour and appends the name of that contour to each
     * to the position.
     */
    using PositionFilter::PositionFilter;

    ~RegionFilter2D();

private:
    // Configurable Interface
    po::options_description options() const override;
    void applyConfiguration(const po::variables_map &vm,
                            const config::OptionTable &config_table) override;

    // Regions
    std::vector<std::string> region_ids_;
    std::vector<std::vector<cv::Point> *> region_contours_;

    /**
     * Check the position to see if it lies within any of the contours defined
     * in the configuration. In the case that the point lies within multiple
     * regions, the first one checked is used and the others are ignored.
     * @param position Position to be filtered
     */
    void filter(oat::Position2D &position) override;
};

}      /* namespace oat */
#endif /* OAT_REGIONFILTER2D_H */
