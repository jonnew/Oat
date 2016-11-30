//******************************************************************************
//* File:   ColorConvert.h
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

#ifndef OAT_COLORCONVERT_H
#define	OAT_COLORCONVERT_H

#include "FrameFilter.h"

namespace oat {

/**
 * A frame color converter
 */
class ColorConvert : public FrameFilter {
public:

    /**
     * @brief Pixel color conversion
     *
     * @param frame_souce_address raw frame source address
     * @param frame_sink_address filtered frame source address
     */
    ColorConvert(const std::string &frame_souce_address,
                 const std::string &frame_sink_address);

    void connectToNode(void) override;
    po::options_description options() const override;
    void applyConfiguration(const po::variables_map &vm,
                            const config::OptionTable &config_table) override;

private:
    void filter(cv::Mat &frame) override;

    int conversion_code_;
    oat::PixelColor color_;
};

}      /* namespace oat */
#endif /* OAT_COLORCONVERT_H */


