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

/**
 * General token recorder
 */
class Recorder {

    //// The controlRecorder routine needs access to Recorder's private members
    friend
    int controlRecorder(std::istream &in,
                        oat::Recorder &recorder,
                        bool print_cmd);
public:

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

    enum ControlMode : int
    {
        NONE = 0,
        LOCAL = 1,
        RPC = 2,
    } control_mode {NONE};
    
    // Source EOF (needed for RecordControl)
    bool source_eof {false};

    // Accessors for control thread
    std::string name(void) const { return name_; }
    bool record_on(void) const { return record_on_; }
    void set_record_on(const bool value) { record_on_ = value; }
    std::string rpc_endpoint(void) const { return rpc_endpoint_; }

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

    // RPC endpoint for interactive control
    std::string rpc_endpoint_ {""};

    // Determines if should file_name be prepended with a timestamp
    bool prepend_timestamp_ {false};

    // True on first file write
    bool files_have_data_ {false};

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
