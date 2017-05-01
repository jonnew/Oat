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

#include <boost/program_options.hpp>

#include "../../lib/base/Component.h"
#include "../../lib/datatypes/Pose.h"
#include "../../lib/shmemdf/Sink2.h"
#include "../../lib/shmemdf/Source.h"

namespace po = boost::program_options;

namespace oat {

class PositionFilter : public Component {

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

    // Component Interface
    oat::ComponentType type(void) const override { return oat::positionfilter; };

protected:
    /**
     * Perform position filtering.
     * @param position Position to be filtered
     */
    virtual void filter(oat::Position2D &position) = 0;

private:
    // Component Interface
    bool connectToNode(void) override;
    int process(void) override;

    // Un-filtered pose SOURCE and filter pose sink
    oat::Source<oat::Pose> position_source_;
    oat::Sink<oat::Pose> position_sink_;

    // Internal, mutable position
    //oat::Position2D internal_position_ {"internal"};

    // Shared position
    //oat::Position2D * shared_position_;

    // Position SINK
    //const std::string position_sink_address_;
};

}      /* namespace oat */
#endif /* OAT_POSITIONFILTER_H */
