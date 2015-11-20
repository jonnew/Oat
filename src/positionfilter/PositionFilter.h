//******************************************************************************
//* File:   PositionFilter.h
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

#ifndef OAT_POSITIONFILTER_H
#define	OAT_POSITIONFILTER_H

#include <string>

#include "../../lib/shmemdf/Source.h"
#include "../../lib/shmemdf/Sink.h"
#include "../../lib/datatypes/Position2D.h"

namespace oat {

/**
 * Abstract position filter.
 * All concrete position filter types implement this ABC.
 */
class PositionFilter {

public:

    /**
     * Abstract position filter.
     * All concrete position filter types implement this ABC.
     * @param position_source_name Un-filtered position SOURCE name
     * @param position_sink_name Filtered position SINK name
     */
    PositionFilter(const std::string &position_source_address,
                   const std::string &position_sink_address);

    virtual ~PositionFilter() { }

    /**
     * PositionFilters must be able to connect to a Source and Sink
     * Nodes in shared memory
     */
    virtual void connectToNode(void);

    /**
     * Obtain un-filtered position from SOURCE. Filter position. Publish filtered
     * position to SINK.
     * @return SOURCE end-of-stream signal. If true, this component should exit.
     */
    bool process(void);

    /**
     * Configure position filter parameters.
     * @param config_file configuration file path
     * @param config_key configuration key
     */
    virtual void configure(const std::string &config_file,
                           const std::string &config_key) = 0;

    // Accessors
    std::string name(void) const { return name_; }

protected:

    /**
     * Perform position filtering.
     * @param position Position to be filtered
     */
    virtual void filter(oat::Position2D &position) = 0;

private:

    // Filter name
    const std::string name_;

    // Un-filtered position SOURCE
    const std::string position_source_address_;
    oat::Source<oat::Position2D> position_source_;

    // Internal, mutable position
    oat::Position2D internal_position_ {"internal"};

    // Shared position
    oat::Position2D * shared_position_;

    // Position SINK
    const std::string position_sink_address_;
    oat::Sink<oat::Position2D> position_sink_;
};

}      /* namespace oat */
#endif /* OAT_POSITIONFILTER_H */

