//******************************************************************************
//* File:   FrameCombiner.cpp
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
#include "../../lib/datatypes/Frame.h"
#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/make_unique.h"

#include "FrameCombiner.h"

namespace oat {

FrameCombiner::FrameCombiner(
                        const std::vector<std::string> &frame_source_addresses,
                        const std::string &frame_sink_address) :
  name_("framcom[" + frame_source_addresses[0] + "...->" + frame_sink_address + "]")
, frame_sink_address_(frame_sink_address)
{
    for (auto &addr : frame_source_addresses) {

        frame_sources_.push_back(
            oat::NamedSource<oat::SharedFrameHeader>(
                addr,
                std::make_unique<oat::Source<oat::SharedFrameHeader>>()
            )
        );
    }
}

void FrameCombiner::connectToNodes() {

    // Touch frame and position source nodes
    for (auto &fs: frame_sources_)
        fs.source->touch(fs.name);

    double sample_rate_hz;
    std::vector<double> all_ts;

    // Connect to frame and position sources
    for (auto &fs: frame_sources_) {
        fs.source->connect();
        all_ts.push_back(fs.source->retrieve().sample().period_sec().count());
    }

    // Examine sample period of sources to make sure they are the same
    if (!oat::checkSamplePeriods(all_ts, sample_rate_hz)) {
        std::cerr << oat::Warn(oat::inconsistentSampleRateWarning(sample_rate_hz));
    }

    // TODO: Type checking for size consistency etc
    // Get frame meta data to format sink
    oat::Source<oat::SharedFrameHeader>::ConnectionParameters param =
            frame_sources_[0].source->parameters();

    // Bind to sink node and create a shared cv::Mat
    frame_sink_.bind(frame_sink_address_, param.bytes);
    shared_frame_ = frame_sink_.retrieve(param.rows, param.cols, param.type);
}

bool FrameCombiner::process() {

    for (fvec_size_t i = 0; i !=  frame_sources_.size(); i++) {

        // START CRITICAL SECTION //
        ////////////////////////////
        if (frame_sources_[i].source->wait() == oat::NodeState::END)
            return true;

        frames_[i] = frame_sources_[i].source->clone();

        frame_sources_[i].source->post();
        ////////////////////////////
        //  END CRITICAL SECTION  //
    }

    combine(frames_, internal_frame_);

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sources to read
    frame_sink_.wait();

    internal_frame_.copyTo(shared_frame_);

    // Tell sources there is new data
    frame_sink_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // Sink was not at END state
    return false;
}

} /* namespace oat */
