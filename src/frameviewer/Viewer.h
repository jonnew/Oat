//******************************************************************************
//* File:   Viewer.h
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
//****************************************************************************

#ifndef OAT_VIEWER_H
#define OAT_VIEWER_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <boost/program_options.hpp>

#include "../../lib/datatypes/Frame.h"
#include "../../lib/shmemdf/Source.h"

namespace po = boost::program_options;

namespace oat {

/**
 * @brief Abstract viewer.
 * All concrete viewer types implement this ABC.
 */
template <typename T>
class Viewer {

    using Clock = std::chrono::high_resolution_clock;
    using Milliseconds = std::chrono::milliseconds;

public:

    /**
     * @brief Abstract viewer.
     * All concrete viewer types implement this ABC.
     * @param source_address Frame SOURCE node address
     */
    explicit Viewer(const std::string &source_name);
    
    virtual ~Viewer();

    /**
     * @brief Append type-specific program options.
     * @param opts Program option description to be specialized.
     */
    virtual void appendOptions(po::options_description &opts) const;

    /**
     * @brief Configure filter parameters.
     * @param vm Previously parsed program option value map.
     */
    virtual void configure(const po::variables_map &vm) = 0;
    /**
     * @brief Connect to SOURCE node from which to get samples
     */
    void connectToNode(void);

    /**
     * @brief Obtain sample from SOURCE and view.
     * @return SOURCE end-of-stream signal. If true, this component should
     * exit.
     */
    bool process(void);

    /**
     * @brief Get viewer name
     * @return name
     */
    std::string name(void) const { return name_; }

protected:

    // Viewer name
    const std::string name_;

    // Source address
    const std::string source_address_;

    // List of allowed configuration options
    std::vector<std::string> config_keys_;

    /**
     * @brief Perform sample display. Override to implement display operation
     * in derived classes.
     * @param Sample to by displayed.
     */
    virtual void display(const T &sample) = 0;

private:

    // Sample SOURCE
    T sample_;
    oat::Source<T> source_;

    // Minimum viewer refresh period
    Clock::time_point tick_, tock_;

    // Display update thread
    std::atomic<bool> running_ {true};
    std::atomic<bool> display_complete_ {true};
    std::mutex display_mutex_;
    std::condition_variable display_cv_;
    std::thread display_thread_;

    // Constants
    const Milliseconds MIN_UPDATE_PERIOD_MS {33};

    /**
     * @brief Asynchronous execution of display(). This function is handled by
     * an asynchronous thread that throttles the GUI update period to
     * MIN_UPDATE_PERIOD_MS.
     */
    void processAsync(void);
};

}      /* namespace oat */
#endif /* OAT_VIEWER_H */
