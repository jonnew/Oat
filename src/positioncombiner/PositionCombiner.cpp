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
#include <thread>
#include <future>

#include "../../lib/shmemdf/Source.h"
#include "../../lib/shmemdf/Sink.h"
#include "../../lib/datatypes/Position2D.h"
#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/make_unique.h"

#include "PositionCombiner.h"

namespace oat {

void PositionCombiner::appendOptions(po::options_description &opts)
{
    // Common program options
    opts.add_options()
        ("config,c", po::value<std::vector<std::string> >()->multitoken(),
        "Configuration file/key pair.\n"
        "e.g. 'config.toml mykey'")
        ;
}

void PositionCombiner::configure(const po::variables_map &vm)
{
    // Pull the sources and sink out as positional options
    auto sources = vm["sources-and-sink"].as< std::vector<std::string> >();

    if (sources.size() < 3)
        throw std::runtime_error("At least two SOURCES and a SINK must be specified.");

    // Last positional argument is the sink.
    auto sink_addr = sources.back();
    sources.pop_back();

    name_ = "posicom[" + sources[0] + "...->" + sink_addr+ "]";

    for (auto &addr : sources) {

        oat::Position2D pos(addr);
        positions_.push_back(std::move(pos));
        position_sources_.push_back(
            oat::NamedSource<oat::Position2D>(
                addr,
                oat::make_unique<oat::Source< oat::Position2D>>()
            )
        );
    }
}

void PositionCombiner::connectToNodes()
{
    // Establish our slot in each node
    for (auto &ps : position_sources_)
        ps.source->touch(ps.name);

    // Examine sample period of sources to make sure they are the same
    double sample_rate_hz;
    std::vector<double> all_ts;

    // Wait for sychronous start with sink when it binds the node
    for (auto &ps : position_sources_) {
        ps.source->connect();
        all_ts.push_back(ps.source->retrieve()->sample_period_sec());
    }

    if (!oat::checkSamplePeriods(all_ts, sample_rate_hz)) {
        std::cerr << oat::Warn(oat::inconsistentSampleRateWarning(sample_rate_hz));
    }

    // Bind to sink node and create a shared position
    position_sink_.bind(position_sink_address_, position_sink_address_);
    shared_position_ = position_sink_.retrieve();
}

bool PositionCombiner::process()
{
    for (pvec_size_t i = 0; i !=  position_sources_.size(); i++) {

        // START CRITICAL SECTION //
        ////////////////////////////
        if (position_sources_[i].source->wait() == oat::NodeState::END)
            return true;

        positions_[i] = position_sources_[i].source->clone();

        position_sources_[i].source->post();
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
