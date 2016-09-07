//******************************************************************************
//* File:   Recorder.cpp
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

#include "Recorder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <mutex>
#include <string>
#include <vector>

#include "../../lib/shmemdf/SharedFrameHeader.h"
#include "../../lib/utility/FileFormat.h"
#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/make_unique.h"

namespace oat {

// TODO: The fact that I'm repeating estentially the same code for two
// different datatypes (Positions and Frames), throughout this class, indicates
// a flaw. The recorder should be capable of writting different incoming data
// types in a more generic fashion. e.g. I should not have to N+1 the code size
// when the recorder becomes capable of handling yet another datatype. It
// should amount to simply creating another Writer deriviative and maybe
// modifying the commandline args. Although, event this last point is suspect.
// Their could be type deduction built into the shared datatypes.

// TODO: Sources should be generic in this case and maybe in general. e.g.
// Filter: T -> F(T) -> T, Source: So -> T, Sink: T -> Si
Recorder::Recorder(const std::vector<std::string> &position_source_addresses,
                   const std::vector<std::string> &frame_source_addresses)
{
    // Start recorder name construction
    name_ = "recorder[" ;

    // Setup position sources
    if (!position_source_addresses.empty()) {

        name_ += position_source_addresses[0];
        if (position_source_addresses.size() > 1)
            name_ += "..";

        for (auto &addr : position_source_addresses) {

            sources_.push_back(
                oat::NamedSource<oat::Position2D>(
                    addr,
                    std::make_unique<oat::Source< oat::Position2D>>()
                    std::make_uqique<oat::PositionWriter>()
                )
            );
        }
    }

    // Setup the frame sources
    if (!frame_source_addresses.empty()) {

        if (!position_source_addresses.empty())
            name_ += ", ";

        name_ += frame_source_addresses[0];
        if (frame_source_addresses.size() > 1)
            name_ += "..";

        for (auto &addr : frame_source_addresses) {

            sources_.push_back(
                oat::NamedSource<oat::Frame>(
                    addr,
                    std::make_unique<oat::Source<oat::Frame>>()
                    std::make_uqique<oat::FrameWriter>()
                )
            );
        }
    }

    name_ +="]";

    // Start the recording tread
    writer_thread_ = std::thread( [this] { writeLoop(); } );
}

Recorder::~Recorder()
{
    // NOTE: The cv::VideoWriter class has internal buffering. Its flushes its
    // buffer on destruction. Because VideoWriter's are accessed via
    // std::unique_ptr, this will happen automatically. However -- don't try to
    // look at a video before the recorder destructs because it will be
    // incomplete! Same with the position file.

    // TODO: Join writer thread
    // Set running to false to trigger thread join
    running_ = false;
    writer_condition_variable_.notify_one();
    writer_thread_.join();
}

void Recorder::connectToNodes()
{
    // Touch frame and position source nodes
    for (auto &s: sources_)
        s.source->touch(s.name);

    //for (auto &ps : position_sources_)
    //    ps.source->touch(ps.name);

    std::vector<double> all_ts;

    // Connect to frame and position sources
    for (auto &s: sources_) {
        fs.source->connect();
        all_ts.push_back(s.source->retrieve().sample().period_sec().count());
    }

    //for (auto &ps : position_sources_) {
    //    ps.source->connect();
    //    all_ts.push_back(ps.source->retrieve()->sample().period_sec().count());
    //}

    // Examine sample period of sources to make sure they are the same
    if (!oat::checkSamplePeriods(all_ts, sample_rate_hz_)) {
        std::cerr << oat::Warn(oat::inconsistentSampleRateWarning(sample_rate_hz_));
    }
}

bool Recorder::writeStreams()
{
    if (record_on_ && initialization_required_) {
        initializeRecording();
        initialization_required_ = false;
    }

    bool source_eof = false;

    // TODO: Read sources
    for (auto &s : sources_) {

         // START CRITICAL SECTION //
        ////////////////////////////
        source_eof_ |= s.source->wait() == oat::NodeState::END;

        // Push newest frame into write queue
        if (record_on_)
            s.writer.push(s.source->clone());

        s.source->post();
        ////////////////////////////
        //  END CRITICAL SECTION  //

    // Read frames
    //int i = 0;
    //for (; i !=  frame_sources_.size(); i++) {

    //     // START CRITICAL SECTION //
    //    ////////////////////////////
    //    source_eof_ |= (frame_sources_[i].source->wait() == oat::NodeState::END);

    //    // Push newest frame into write queue
    //    if (record_on_)
    //        writers_[i]->push(frame_sources_[i].source->clone());

    //    frame_sources_[i].source->post();
    //    ////////////////////////////
    //    //  END CRITICAL SECTION  //
    //}

    //// Read positions
    //for (; i !=  position_sources_.size(); i++) {

    //    // START CRITICAL SECTION //
    //    ////////////////////////////
    //    source_eof_ |= (position_sources_[i].source->wait() == oat::NodeState::END);

    //    // Push newest position into write queue
    //    if (record_on_)
    //        writers_[i]->push(position_sources_[i].source->clone());

    //    position_sources_[i].source->post();
    //    ////////////////////////////
    //    //  END CRITICAL SECTION  //
    //}

    // Notify the writer thread that there are new queued samples
    writer_condition_variable_.notify_one();

    return source_eof_;
}

void Recorder::writeLoop()
{
    while (running_) {

        std::unique_lock<std::mutex> lk(writer_mutex_);
        writer_condition_variable_.wait_for(lk, std::chrono::milliseconds(10));

        //std::cout << "Dummy write.\n";
        for (auto &s: sources_)
            s.writer->write();
    }
}

// TODO: clone()'s below are not thread safe
void Recorder::initializeRecording()
{
    std::string timestamp = oat::createTimeStamp();

    // TODO: package writer along with its source using a tuple or something.
    // You are relying way to much on proper indexing here

    // Create a writer for each frame source
    for (auto &s : sources_) {

        std::string file_path = generateFileName(timestamp, s.name, ".avi");
        //writers_.push_back(std::make_unique<oat::WriterBase>(
        //    std::in_place<oat::FrameWriter>(), file_path)
        //);
        s.writers = std::make_unique<oat::WritreBase>(std::in_place<oat::FrameWriter>(), file_path);
        frame_writers_.back()->initialize(s.name, s.source->clone());
    }

    //// Create a writer for each frame source
    //for (auto &s : frame_sources_) {

    //    std::string file_path = generateFileName(timestamp, s.name, ".avi");
    //    //writers_.push_back(std::make_unique<oat::WriterBase>(
    //    //    std::in_place<oat::FrameWriter>(), file_path)
    //    //);
    //    writers_.emplace_back(std::in_place<oat::FrameWriter>(), file_path);
    //    frame_writers_.back()->initialize(s.name, s.source->clone());
    //}

    //// Create a writer for each position source
    //for (auto &p : position_sources_) {

    //    std::string file_path = generateFileName(timestamp, p.name, ".json");
    //    //writers_.push_back(std::make_unique<oat::WriterBase>(
    //    //    std::in_place<oat::PositionWriter>(), file_path)
    //    //);
    //    writers_.emplace_back(std::in_place<oat::PositionWriter>(), file_path);
    //    position_writers_.back()->initialize(p.name, p.source->clone());

    //    // TODO: Hack.
    //    position_writers_.back()->set_verbose_file(verbose_file_);
    //}

}

/**
 * @brief Generate unified file name for all streams.
 * @return Recording path.
 */
std::string Recorder::generateFileName(const std::string timestamp,
                                       const std::string &source_name,
                                       const std::string &extension)
{
    std::string base_fid = source_name;
    if (!file_name_.empty())
        base_fid += "_" + file_name_;

    std::string full_path;
    int err = oat::createSavePath(full_path,
            save_path_,
            base_fid + extension,
            timestamp + "_",
            prepend_timestamp_);

    if (err) {
        throw std::runtime_error("Recording file initialization exited "
                                 "with error " + std::to_string(err));
    }

    if (!allow_overwrite_)
       oat::ensureUniquePath(full_path);

    return full_path;
}

} /* namespace oat */
