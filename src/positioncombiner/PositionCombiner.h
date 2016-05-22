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

#ifndef OAT_POSITIONCOMBINER_H
#define	OAT_POSITIONCOMBINER_H

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
 * Abstract position combiner.
 * All concrete position combiner types implement this ABC.
 */
class PositionCombiner {

public:

    using pvec_size_t = oat::NamedSourceList<oat::Position2D>::size_type;

    /**
     * Abstract position combiner.
     * All concrete position combiner types implement this ABC.
     * @param position_source_addresses A vector of position SOURCE addresses
     * @param position_sink_address Combined position SINK address
     */
    PositionCombiner(const std::vector<std::string> &position_source_addresses,
                     const std::string &position_sink_address);

    /**
     * Position combiner SOURCEs must be able to connect to a NODEs from
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
     * Perform position combination.
     * @param sources SOURCE position servers
     * @return combined position
     */
    virtual void combine(const std::vector<oat::Position2D>& source_positions,
                         oat::Position2D &combined_position) = 0;

    /**
     * Get the number of SOURCE positions.
     * @return number of SOURCE positions
     */
    int num_sources(void) const {return position_sources_.size(); };

private:

    // Combiner name
    std::string name_;

    // Position SOURCES object for un-combined positions
    std::vector<oat::Position2D> positions_;
    oat::NamedSourceList<oat::Position2D> position_sources_;

    // Combined position
    oat::Position2D internal_position_ {"internal"};

    // Position SINK object for publishing combined position
    oat::Position2D * shared_position_ {nullptr};
    const std::string position_sink_address_;
    oat::Sink<oat::Position2D> position_sink_;
};

}      /* namespace oat */
#endif	/* OAT_POSITIONCOMBINER_H */

