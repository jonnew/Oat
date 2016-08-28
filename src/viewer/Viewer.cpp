//******************************************************************************
//* File:   Viewer.cpp
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

#include "Viewer.h"

#include <chrono>
#include <future>
#include <iostream>
#include <string>

#include "../../lib/shmemdf/Source.h"

namespace oat {

template<typename T>
Viewer<T>::Viewer(const std::string &source_address) :
  name_("viewer[" + source_address + "]")
, source_address_(source_address)
{
    // Initialize GUI update timers
    tick_ = Clock::now();
    tock_ = Clock::now();

    // Start display thread
    display_thread_ = std::thread( [this] {processAsync();} );
}

template<typename T>
Viewer<T>::~Viewer()
{
    running_ = false;
    display_cv_.notify_one();
    display_thread_.join();
}

template<typename T>
void Viewer<T>::appendOptions(po::options_description &opts) {

    // Common program options
    opts.add_options()
        ("config,c", po::value<std::vector<std::string> >()->multitoken(),
        "Configuration file/key pair.\n"
        "e.g. 'config.toml mykey'")
        ;
}

template<typename T>
void Viewer<T>::connectToNode() {

    // Establish our a slot in the node
    source_.touch(source_address_);

    // Wait for synchronous start with sink when it binds the node
    source_.connect();
}

template<typename T>
bool Viewer<T>::process() {

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sink to write to node
    if (source_.wait() == oat::NodeState::END)
        return true;

    // Clone the shared frame
    source_.copyTo(sample_);

    // Tell sink it can continue
    source_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // Get current time
    tick_ = Clock::now();

    // Figure out the time since we last updated the viewer
    Milliseconds duration =
        std::chrono::duration_cast<Milliseconds>(tick_ - tock_);

    // If the minimum update period has passed, and display thread is not busy,
    // show the new sample on the display thread. This prevents GUI updates
    // from holding up more important upstream processing.
    if (duration > MIN_UPDATE_PERIOD_MS && display_complete_)
        display_cv_.notify_one();

    // Sink was not at END state
    return false;
}

template<typename T>
void Viewer<T>::processAsync() {

    while (running_) {

        std::unique_lock<std::mutex> lk(display_mutex_);
        display_cv_.wait(lk);

        // Prevent desctructor from calling display() after derived class has
        // been desctructed
        if (!running_)
            break;

        display_complete_ = false;
        display(sample_); // Implemented in concrete class
        tock_ = Clock::now();
        display_complete_ = true;
    }
}

// Explicit instantiations
template class oat::Viewer<oat::Frame>;

} /* namespace oat */
