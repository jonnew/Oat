//******************************************************************************
//* File:   Component.cpp
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

#include "Component.h"
#include "Globals.h"

#include <chrono>
#include <exception>
#include <iostream>
#include <sstream>
#include <thread>

#include <boost/interprocess/exceptions.hpp>

#include "../../lib/utility/ZMQHelpers.h"

namespace oat {

// Global via extern in Globals.h
volatile sig_atomic_t quit = 0;

// Signal handler to ensure shared resources are cleaned on exit due to ctrl-c
static void sigHandler(int)
{
    quit = 1;
}

Component::Component()
{
    // Install Ctrl-c signal handler
    std::signal(SIGINT, sigHandler);
}

void Component::run()
{
    // Loop until quit
    runComponent();
}

void Component::runComponent()
{
    try {

        if (!connectToNode())
            return;

        bool end_of_stream = false;
        while (!end_of_stream && !quit) {
            end_of_stream = process();
        }

    } catch (const boost::interprocess::interprocess_exception &ex) {

        // Error code 1 indicates a SIGINT during a call to wait(),
        // which is normal behavior
        if (ex.get_error_code() != 1)
            throw;
    }
}

} /* namespace oat */
