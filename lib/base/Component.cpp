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

#include <chrono>
#include <exception>
#include <iostream>
#include <sstream>
#include <thread>

#include <boost/interprocess/exceptions.hpp>

#include "../../lib/utility/ZMQHelpers.h"

namespace oat {

static std::exception_ptr ctrl_ex = nullptr;

// Signal handler to ensure shared resources are cleaned on exit due to ctrl-c
static void sigHandler(int)
{
    quit = 1;
}

static void catchSignals (void)
{
    struct sigaction action;
    action.sa_handler = sigHandler;
    action.sa_flags = 0;
    sigemptyset (&action.sa_mask);
    sigaction (SIGINT, &action, NULL);
    sigaction (SIGTERM, &action, NULL);
}

Component::Component()
{
    catchSignals();
}

void Component::identity(char *id, const size_t n) const
{
    auto tid = std::this_thread::get_id();
    std::stringstream ping_msg;
    ping_msg << std::hex << std::uppercase;
    ping_msg << "OAT"; // Must not start with binary 0
    ping_msg << "/";
    ping_msg << tid;
    std::strncpy(id, ping_msg.str().data(), n);
}

void Component::run()
{
    // Get endpoint from program options
    zmq::context_t ctx(1);
    auto control_thread = std::thread( [this, &ctx] { runController(ctx); } );

    // Loop until quit
    runComponent();

    // Unblock zmq recv and let loop respect quit
    std::cout << "Exiting." << std::endl;
    ctx.close();

    // Join processing and control threads, in possible
    if (control_thread.joinable())
        control_thread.join();

    // If an exception occured in control thread, rethrow it on the joined
    // thread
    if (ctrl_ex) {
        std::rethrow_exception(ctrl_ex);
    }
}

void Component::runComponent()
{
    try {

        connectToNode();

        while (!quit)
            quit = process();

    } catch (const boost::interprocess::interprocess_exception &ex) {

        // Error code 1 indicates a SIGINT during a call to wait(),
        // which is normal behavior
        if (ex.get_error_code() != 1)
            throw;
    }
}

void Component::runController(zmq::context_t &context, const char *endpoint)
{
    try {

        auto ctrl_socket = getCtrlSocket(context, endpoint);

        //  Configure socket to not wait at close time
        //std::cout << "Restarting component control socket." << std::endl;

        // Construct ping message: type/name
        std::stringstream ping_msg;
        ping_msg << std::to_string(static_cast<uint16_t>(type()));
        ping_msg << "/";
        ping_msg << name();

        int retries_left = REQUEST_RETRIES; 

        //oat::sendStringMore(ctrl_socket, ""); // Delimeter
        oat::sendString(ctrl_socket, ping_msg.str());
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        // Execute control loop
        while (!quit) {

            bool expect_reply = true;
            while (expect_reply) {

                // Poll the socket and find all existing connections
                zmq::pollitem_t p[] = {{*ctrl_socket, 0, ZMQ_POLLIN, 0}};
                zmq::poll(&p[0], 1, REQUEST_TIMEOUT_MS); 

                if (p[0].revents & ZMQ_POLLIN && !quit) {

                    //oat::recvString(ctrl_socket); // Delimeter
                    auto reply = oat::recvString(ctrl_socket);

                    //std::cout << "Got reply: " << reply << std::endl;
                    // Interpret message
                    if (reply != "ACK") {
                        quit = control(reply.c_str());
                    }

                    //oat::sendStringMore(ctrl_socket, ""); // Delimeter
                    oat::sendString(ctrl_socket, ping_msg.str());
                    retries_left = REQUEST_RETRIES;
                    expect_reply = false;

                } else if (retries_left-- == 0) {
                    expect_reply = false;
                    quit = 1;
                    break;
                } else {
                    //std::cout << "Client restart." << std::endl;
                    delete ctrl_socket;
                    ctrl_socket = getCtrlSocket(context, endpoint);
                    oat::sendString(ctrl_socket, ping_msg.str());
                }
            }
        }

        delete ctrl_socket;

    } catch (zmq::error_t &ex) {

        // ETERM occurs during interrupt from ctrl-c
        if (ex.num() != ETERM) {
            ctrl_ex = std::current_exception();
            quit = 1; // Break processing loop
        }
    }
}

zmq::socket_t *Component::getCtrlSocket(zmq::context_t &context,
                                        const char *endpoint)
{
    zmq::socket_t *socket = new zmq::socket_t(context, ZMQ_REQ);
    char id[32];
    identity(id, 32);
    socket->setsockopt(ZMQ_IDENTITY, id, std::strlen(id));
    socket->connect(endpoint);

    //  Configure socket to not wait at close time
    socket->setsockopt(ZMQ_LINGER, 0);
    return socket;
}

} /* namespace oat */
