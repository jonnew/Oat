//******************************************************************************
//* File:   MaskSubtractor.h
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

#ifndef OAT_MASKSUBTRACTOR_H
#define	OAT_MASKSUBTRACTOR_H

#include <string>
#include <vector>

#include "FrameCombiner.h"

namespace oat {

enum class MaskType {
    AND = 0,
    OR = 1,
    XOR = 2,
};

/**
 * A dynamic frame masker.
 */
class MaskSubtractor : public FrameCombiner {
public:

    /**
     * A dynamic frame masker.
     * @brief TODO
     * @param position_source_addresses A vector of position SOURCE addresses
     * @param position_sink_address Combined position SINK address
     */
    MaskSubtractor(const std::vector<std::string> &frame_source_addresses,
                   const std::string &frame_sink_address);

    void configure(const std::string &config_file,
                   const std::string &config_key) override;

private:

    void combine(const std::vector<oat::Frame>& source_frames,
                 oat::Frame &combined_frame) override;

    /// SOURCE frame to apply masks to
    int64_t masked_frame_idx_ {0};

    /// Mask combination operation for multiple mask sources
    oat::MaskType mask_type {oat::MaskType::AND};
};

}      /* namespace oat */
#endif /* OAT_MASKSUBTRACTOR_H */
