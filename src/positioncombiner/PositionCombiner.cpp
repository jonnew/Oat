//******************************************************************************
//* File:   PositionCombiner.cpp
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

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "../../lib/shmemdf/Source.h"
#include "../../lib/shmemdf/Sink.h"
#include "../../lib/datatypes/Position2D.h"
#include "../../lib/utility/make_unique.h"

#include "PositionCombiner.h"

namespace oat {

PositionCombiner::PositionCombiner(
                        const std::vector<std::string> &position_source_addresses,
                        const std::string &position_sink_address) :
      name_("posicom[" + position_source_addresses[0] + "...->" + position_sink_address + "]")
    , position_sink_address_(position_sink_address)
{

    for (auto &addr : position_source_addresses) {

        oat::Position2D pos(addr);
        positions_.push_back(std::move(pos));
        position_sources_.push_back(std::make_pair(addr,
                std::make_unique<oat::Source< oat::Position2D >>() ));
    }
}

PositionCombiner::~PositionCombiner() {

//    // Delete the memory pointed to by `new oat::Source<oat::Position2D>()`
//    for (auto &pos : position_sources_)
//        delete pos.second;
}

void PositionCombiner::connectToNodes() {

    // Connect to position source nodes
    for (auto &pos : position_sources_)
        pos.second->connect(pos.first);

    // Bind to sink node and create a shared position
    position_sink_.bind(position_sink_address_);
    shared_position_ = position_sink_.retrieve();
}

bool PositionCombiner::process() {

    for (pvec_size_t i = 0; i !=  position_sources_.size(); i++) {

        // START CRITICAL SECTION //
        ////////////////////////////
        if (position_sources_[i].second->wait() == oat::NodeState::END)
            return true;

        positions_[i] = position_sources_[i].second->clone();

        position_sources_[i].second->post();
        ////////////////////////////
        //  END CRITICAL SECTION  //
    }

    combine(positions_, internal_position_);

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sources to read
    position_sink_.wait();

    *shared_position_ = internal_position_;

    // Tell sources there is new data
    position_sink_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // Sink was not at END state
    return false;
}

} /* namespace oat */