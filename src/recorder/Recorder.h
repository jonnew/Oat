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

#include "../../lib/base/ControllableComponent.h"
#include "../../lib/base/Configurable.h"

namespace oat {
namespace po = boost::program_options;

class Recorder : public ControllableComponent, public Configurable<true> {

public:
    ~Recorder();

    // Implement ControllableComponent interface
    oat::ComponentType type(void) const override { return oat::recorder; };
    std::string name(void) const override { return name_; }

private:
    // Implement ControllableComponent interface
    bool connectToNode(void) override;
    int process(void) override;
    void applyCommand(const std::string &command) override;
    oat::CommandDescription commands(void) override;

    // Implement Configurable interface
    po::options_description options() const override;
    void applyConfiguration(const po::variables_map &vm,
                            const config::OptionTable &config_table) override;

    /**
     * @brief Create and initialize recording file(s). Must be called
     * before process().
     */
    void initializeRecording(void);

    // Name of this recorder
    std::string name_;

    // Recorder in running state (i.e. all threads should remain responsive for
    // new data coming down the pipeline)
    std::atomic<bool> running_ {true};

    // Recording gate can be toggled on and off interactively from other
    // threads and processes
    std::atomic<bool> record_on_ {true};

    // Sample rate of this recorder
    // The true sample rate is enforced by the slowest SOURCE since all SOURCEs
    // are synchronized. User will be warned if SOURCE sample rates differ.
    double sample_rate_hz_ {0.0};

    // Folder in which files will be saved
    std::string save_path_ {"."};

    // Base file name
    std::string file_name_ {""};

    // Determines if should file_name be prepended with a timestamp
    bool prepend_timestamp_ {false};

    // True on first file write
    bool files_have_data_ {false};

    // Executed by writer_thread_
    void writeLoop(void);

    // Writers (each owns its SOURCE)
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
