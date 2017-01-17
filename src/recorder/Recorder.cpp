//******************************************************************************
//* File:   Recorder.cpp
//* Author: Jon Newman <jpnewman snail mit dot edu>
//*
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu)
//* All right reserved.
//* This file is part of the Oat projecw->
//* This is free software: you can redistribute it and/or modify
//* it under the terms of the GNU General Public License as published by
//* the Free Software Foundation, either version 3 of the License, or
//* (at your option) any later version.
//* This software is distributed in the hope that it will be useful,
//* but WITHOUT ANY WARRANTY; without even the implied warranty of
//* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//* GNU General Public License for more details.
//* You should have received a copy of the GNU General Public License
//* along with this source code.  If not, see <http://www->gnu.org/licenses/>.
//******************************************************************************

#include "Recorder.h"
#include "Writer.h"
#include "FrameWriter.h"
#include "PositionWriter.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <mutex>
#include <string>
#include <vector>

#include "../../lib/utility/FileFormat.h"
#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/make_unique.h"
#include "../../lib/shmemdf/Helpers.h"

namespace oat {

Recorder::~Recorder()
{
    // Set running to false to trigger thread join
    running_ = false;
    writer_condition_variable_.notify_one();
    if (writer_thread_.joinable())
        writer_thread_.join();

    // If files were never written to, get rid of them
    if (!files_have_data_) {
        for (auto &w : writers_)
            w->deleteFile();
    }
}

po::options_description Recorder::options() const
{
    // Update CLI options
    po::options_description local_opts;
    local_opts.add_options()
        ("frame-sources,s", po::value< std::vector<std::string> >()->multitoken(),
        "The names of the FRAME SOURCES that supply images to save to video.")
        ("position-sources,p", po::value< std::vector<std::string> >()->multitoken(),
        "The names of the POSITION SOURCES that supply object positions "
        "to be recorded.")
        ("filename,n", po::value<std::string>(),
        "The base file name. If not specified, defaults to the SOURCE "
        "name.")
        ("folder,f", po::value<std::string>(),
        "The path to the folder to which the video stream and position "
        "data will be saved. If not specified, defaults to the "
        "current directory.")
        ("date,d",
        "If specified, YYYY-MM-DD-hh-mm-ss_ will be prepended to the "
        "filename.")
        ("allow-overwrite,o",
        "If set and save path matches and existing file, the file will "
        "be overwritten instead of a incremental numerical index being "
        "appended to the file name.")
        ("fourcc,F", po::value<std::string>(),
         "Four character code (https://en.wikipedia.org/wiki/FourCC) used to "
         "specify the codec used for AVI video compression. Must be specified as "
         "a 4-character string (see http://www.fourcc.org/codecs.php for "
         "possible options). Not all valid FOURCC codes will work: it "
         "must be implemented by the low  level writer. Common values are "
         "'DIVX' or 'H264'. Defaults to 'None' indicating uncompressed "
         "video.")
        ("binary-file,b",
         "Position data will be written as numpy data file (version 1.0) "
         "instead of JSON. Each position data point occupies a single entry "
         "in a structured numpy array. Individual position characteristics "
         "are described in the arrays dtype.")
        ("concise-file,c",
         "If set and using JSON file format, indeterminate position data fields "
         "will not be written e.g. pos_xy will not be written even when "
         "pos_ok = false. This means that position objects will be of "
         "variable size depending on the validity of whether a position was "
         "detected or not, potentially complicating file parsing.")
        ;

    return local_opts;
}

void Recorder::applyConfiguration(const po::variables_map &vm,
                                  const config::OptionTable &config_table)
{
    // Sources
    if (vm.count("frame-sources")) {

        auto addrs = vm["frame-sources"].as<
            std::vector<std::string> >();

        oat::config::checkForDuplicateSources(addrs);

        for (auto &a : addrs)
            writers_.emplace_back(oat::make_unique<FrameWriter>(a));
    }

    if (vm.count("position-sources")) {

        auto addrs = vm["position-sources"].as<
            std::vector<std::string> >();

        oat::config::checkForDuplicateSources(addrs);

        for (auto &a : addrs)
            writers_.emplace_back(oat::make_unique<PositionWriter>(a));
    }

    if (writers_.size() == 0)
        throw std::runtime_error("At least one SOURCE must be provided.");
    else if (writers_.size() > 1)
        name_ = "recorder[" + writers_[0]->addr() + "..]";
    else
        name_ = "recorder[" + writers_[0]->addr()  + "]";

    // Base file name
    oat::config::getValue(vm, config_table, "filename", file_name_);

    // Save folder
    oat::config::getValue(vm, config_table, "folder", save_path_);

    // Date
    oat::config::getValue(vm, config_table, "date", prepend_timestamp_);

    // Writer specific options
    for (auto &w : writers_)
        w->configure(config_table, vm);

    // Start the recording thread
    writer_thread_ = std::thread( [this] { writeLoop(); } );
}

bool Recorder::connectToNode()
{
    // Touch frame and position source nodes
    for (auto &w : writers_)
        w->touch();

    std::vector<double> all_ts;
    for (auto &w : writers_) {
        if (w->connect() != SourceState::CONNECTED)
            return false;
        all_ts.push_back(w->sample_period_sec());
    }

    // Examine sample period of sources to make sure they are the same
    if (!oat::checkSamplePeriods(all_ts, sample_rate_hz_))
        std::cerr << oat::Warn(oat::inconsistentSampleRateWarning(sample_rate_hz_));

    // Setup file, etc
    initializeRecording();

    return true;
}

int Recorder::process()
{
    bool source_eof = false;

    // Read sources, push samples to write buffers
    for (auto &w : writers_) {

        // START CRITICAL SECTION //
        ////////////////////////////
        source_eof |= w->wait() == oat::NodeState::END;

        if (record_on_ && !source_eof) {
           w->push();
           files_have_data_ = true;
        }

        w->post();
        ////////////////////////////
        //  END CRITICAL SECTION  //
    }

    // Notify the writer thread that there are new queued samples
    writer_condition_variable_.notify_one();

    return source_eof;
}

oat::CommandDescription Recorder::commands()
{
    const oat::CommandDescription commands{
        {"start", "Start recording. This will append the file if it "
                  "already exists. It will create a new one if it doesn't." },
        {"pause", "Pause recording. This will pause the recording "
                  "without creating a new file." },
        {"new", "Start a new file using folder location and file name "
                "options as provided in command line arguements."},
    };

    return commands;
}

void Recorder::applyCommand(const std::string &command)
{
    const auto cmds = commands();

    if (command == "start") {
        record_on_ = true;
    } else if (command == "pause") {
        record_on_ = false;
    } else if (command == "new") {
        // TODO: makeNewFile()
        std::cout << "did not implement 'new' yet...\n";
    }
}

void Recorder::writeLoop()
{
    while (running_) {

        std::unique_lock<std::mutex> lk(writer_mutex_);
        writer_condition_variable_.wait_for(lk, std::chrono::milliseconds(10));

        for (auto &w : writers_)
            w->write();
    }
}

void Recorder::initializeRecording()
{
    std::string timestamp = oat::createTimeStamp();

    for (auto &w : writers_) {
        auto fid = generateFileName(timestamp, w->addr());
        w->initialize(fid);
    }
}

std::string Recorder::generateFileName(const std::string timestamp,
                                       const std::string &source_name)
{
    std::string base_fid = source_name;
    if (!file_name_.empty())
        base_fid += "_" + file_name_;

    std::string full_path;
    int err = oat::createSavePath(
        full_path, save_path_, base_fid, timestamp + "_", prepend_timestamp_);

    if (err == 1)
        throw std::runtime_error("Requested save path does not exist.");
    if (err == 3)
        throw std::runtime_error("File name empty.");

    return full_path;
}

} /* namespace oat */
