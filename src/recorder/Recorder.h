//******************************************************************************
//* File:   Recorder.h
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
//*****************************************************************************

#ifndef OAT_RECORDER_H
#define OAT_RECORDER_H

#include "Writer.h"

#include <boost/program_options.hpp>

#include <atomic>
#include <condition_variable>
#include <memory>
#include <string>
#include <thread>

namespace oat {
namespace po = boost::program_options;

//// TODO: Move to datatypes if you are going to use
//enum class TokenType : int {
//    Any = -1,
//    Frame,
//    Position
//};

/**
 * Position and frame recorder.
 */
class Recorder {

    //// The controlRecorder routine needs access to Recorder's private members
    //friend
    //int controlRecorder(std::istream &in,
    //                    oat::Recorder &recorder,
    //                    bool print_cmd);
public:
    /**
     * Position and frame recorder.
     * @param position_source_addresses Addresses specifying position SOURCES
     * to record
     * @param frame_source_addresses Addresses specifying frame SOURCES to
     * record
     */
    //Recorder(const std::vector<std::string> &position_source_addresses,
    //         const std::vector<std::string> &frame_source_addresses);

    ~Recorder();

    /**
     * @brief Append type-specific program options.
     * @param opts Program option description to be specialized.
     */
    virtual void appendOptions(po::options_description &opts);

    /**
     * @brief Configure component parameters.
     * @param vm Previously parsed program option value map.
     */
    virtual void configure(const po::variables_map &vm);

    /**
     * Recorder SOURCEs must be able to connect to a NODEs from
     * which to receive positions and frames.
     */
    void connectToNodes(void);

    /**
     * @brief Create and initialize recording file(s). Must be called
     * before writeStreams.
     *
     */
    void initializeRecording(void);

    /** Collect frames and positions from SOURCES. Write frames and positions
     * to file.
     * @return SOURCE end-of-stream signal. If true, this component should
     * exit.
     */
    bool writeStreams(void);

    // Accessors for control thread
    std::string name(void) { return name_; }
    bool record_on(void) const { return record_on_; }
    void set_record_on(const bool value) { record_on_ = value; }
    //void set_save_path(const std::string &value) { save_path_ = value; }
    //void set_file_name(const std::string &value) { file_name_ = value; }
    //
protected:
    // List of allowed configuration options
    std::vector<std::string> config_keys_;

private:

    // Name of this recorder
    std::string name_;

    // Recorder in running state (i.e. all threads should remain responsive
    // for new data coming down the pipeline)
    std::atomic<bool> running_ {true};

    // Recording gate can be toggled on and off interactively from other
    // threads and processes
    std::atomic<bool> record_on_ {true};

    // Sample rate of this recorder
    // The true sample rate is enforced by the slowest SOURCE since all SOURCEs
    // are sychronized. User will be warned if SOURCE sample rates differ.
    double sample_rate_hz_ {0.0};

    // Folder in which files will be saved
    std::string save_path_ {"."};

    // Base file name
    std::string file_name_ {""};

    // Determines if should file_name be prepended with a timestamp
    bool prepend_timestamp_ {false};

    // Files must be initialized before first write
    bool initialization_required_ {true};

    // Executed by writer_thread_
    void writeLoop(void);

    std::vector<std::string> source_addrs_;

    // Writers (own sources)
    std::vector<std::unique_ptr<Writer>> writers_;

    // File-writer threading
    std::thread writer_thread_;
    std::mutex writer_mutex_;
    std::condition_variable writer_condition_variable_;

    // Create file name from components
    std::string generateFileName(const std::string timestamp,
                                 const std::string &source_name);
};

}      /* namespace oat */
#endif /* OAT_RECORDER_H */
