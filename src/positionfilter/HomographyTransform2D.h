//******************************************************************************
//* File:   HomographyTransform2D.h
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

#ifndef OAT_HOMGRAPHICTRANSFORM2D_H
#define	OAT_HOMGRAPHICTRANSFORM2D_H

#include "PositionFilter.h"

#include <string>
#include <opencv2/core/mat.hpp>

namespace oat {

class HomographyTransform2D : public PositionFilter {

public:
    /**
     * A 2D homography.
     * Transform position units. Typically used to perform projective
     * transformation to map pixels coordinates to world coordinates.
     */
    using PositionFilter::PositionFilter;

private:
    // Configurable Interface
    po::options_description options() const override;
    void applyConfiguration(const po::variables_map &vm,
                            const config::OptionTable &config_table) override;

    // 2D homography matrix
    //bool homography_valid_ {false};
    cv::Matx33d homography_ {1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0};

    /**
     * Apply homography transform.
     * @param Position to be projected
     */
    void filter(oat::Position2D& position) override;
};

}      /* namespace oat */
#endif /* OAT_HOMGRAPHICTRANSFORM2D_H */
