//******************************************************************************
//* File:   PositionCombiner.h
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

#ifndef OAT_FRAMECOMBINER_H
#define	OAT_FRAMECOMBINER_H

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "../../lib/shmemdf/Helpers.h"
#include "../../lib/shmemdf/Source.h"
#include "../../lib/shmemdf/Sink.h"
#include "../../lib/datatypes/Position2D.h"

namespace oat {

/**
 * Abstract frame combiner.
 * All concrete frame combiner types implement this ABC.
 */
class FrameCombiner {

public:

    using fvec_size_t =
        std::vector<oat::NamedSource<oat::SharedFrameHeader>>::size_type;

    /**
     * Abstract frame combiner.
     * All concrete frame combiner types implement this ABC.
     * @param frame_source_addresses A vector of frame SOURCE addresses
     * @param frame_sink_address Combined frame SINK address
     */
    FrameCombiner(const std::vector<std::string> &frame_source_addresses,
                  const std::string &frame_sink_address);

    /**
     * Frame combiner SOURCEs must be able to connect to a NODEs from
     * which to receive positions and a SINK to send combined positions.
     */
    virtual void connectToNodes(void);

    /**
     * Obtain positions from all SOURCES. Combine positions. Publish combined position
     * to SINK.
     * @return SOURCE end-of-stream signal. If true, this component should exit.
     * TODO: check that position length units are the same before combination
     */
    bool process(void);

    std::string name(void) const { return name_; }

    /**
     * Configure position combiner parameters.
     * @param config_file configuration file path
     * @param config_key configuration key
     */
    virtual void configure(const std::string &config_file,
                           const std::string &config_key) = 0;

protected:

    /** 
     * @brief Perform frame combination
     * 
     * @param source_frame SOURCE frame servers
     * @param frame_position 
     */
    virtual void combine(const std::vector<oat::Frame> &source_frames,
                         oat::Frame &combined_frame) = 0;

    /**
     * Get the number of SOURCE frames.
     * @return number of SOURCE frames
     */
    int num_sources(void) const {return frame_sources_.size(); };

private:

    // Combiner name
    std::string name_;

    // Position SOURCES for un-combined framea
    std::vector<oat::Frame> frames_;
    oat::NamedSourceList<oat::SharedFrameHeader> frame_sources_;

    // Currently processed frame
    oat::Frame internal_frame_;

    // Frame sink
    const std::string frame_sink_address_;
    oat::Sink<oat::SharedFrameHeader> frame_sink_;

    // Currently acquired, shared frame
    oat::Frame shared_frame_;
};

}      /* namespace oat */
#endif /* OAT_FRAMECOMBINER_H */

