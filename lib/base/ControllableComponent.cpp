//******************************************************************************
//* File:   ControllableComponent.cpp
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

#include "ControllableComponent.h"
#include "Globals.h"

#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

#include <boost/interprocess/exceptions.hpp>

#include "../../lib/utility/ZMQHelpers.h"

namespace oat {

// For passing exceptions from control thread
static std::exception_ptr ctrl_ex = nullptr;

void ControllableComponent::identity(char *id, const size_t n) const
{
    std::stringstream ping_msg;
    ping_msg << std::hex << std::uppercase;
    ping_msg << "OAT"; // Must not start with binary 0, ZMQ rule
    ping_msg << "/";
    ping_msg << std::this_thread::get_id();
    std::strncpy(id, ping_msg.str().data(), n);
}

void ControllableComponent::run()
{
    // Get endpoint from program options
    //zmq::context_t ctx(1);
    auto control_thread = std::thread([this] { runController(); });

    // Loop until quit
    runComponent();

    // Join processing and control threads, in possible
    if (control_thread.joinable())
        control_thread.join();

    // If an exception occured in control thread, rethrow it on the joined
    // thread
    if (ctrl_ex)
        std::rethrow_exception(ctrl_ex);
}

void ControllableComponent::runController(const char *endpoint)
{
    zmq::context_t ctx(1);

    try {

        auto ctrl_socket = getCtrlSocket(ctx, endpoint);

        oat::sendStringMore(ctrl_socket, ""); // Delimeter
        oat::sendString(ctrl_socket, whoAmI());

        // Execute control loop
        while (!quit) {

            // Poll the socket and find all existing connections
            zmq::pollitem_t p[] = {{*ctrl_socket, 0, ZMQ_POLLIN, 0}};
            zmq::poll(&p[0], 1, REQUEST_TIMEOUT_MS);

            if (p[0].revents & ZMQ_POLLIN && !quit) {

                // Found a command, run it.
                oat::recvString(ctrl_socket); // Delimeter
                auto command = oat::recvString(ctrl_socket);
                quit = control(command);

            } else {
                // If we did not get a reply on this socket
                // REQUEST_TIMEOUT_MS, tear it down and make a new one
                delete ctrl_socket;
                ctrl_socket = getCtrlSocket(ctx, endpoint);
                oat::sendStringMore(ctrl_socket, ""); // Delimeter
                oat::sendString(ctrl_socket, whoAmI());
            }
        }

        delete ctrl_socket;

    } catch (zmq::error_t &ex) {

        // ETERM occurs during interrupt from ctrl-c, otherwise pass exception
        // to processing thread
        if (ex.num() != ETERM)
            ctrl_ex = std::current_exception();
    }

    ctx.close();
}

int ControllableComponent::control(const std::string &command)
{
    if (command == "quit" || command == "Quit") {
        return 1;
    } else {

        // Check that command is in hash
        if (commands().count(command))
            applyCommand(command);
    }

    return 0;
}

std::string ControllableComponent::whoAmI()
{
    // JSON string with name, type, and command/description map
    std::stringstream whoami;
    whoami << "{";
    whoami << "\"name\":\"" << name() << "\",";
    whoami << "\"type\":" << std::to_string(static_cast<uint16_t>(type())) << ",";

    auto cmds = commands();
    if (!cmds.empty()) {

        whoami << "\"commands\":{";
        for (const auto &c : cmds)
            whoami << "\"" << c.first << "\":\"" << c.second << "\",";
        whoami.seekp(-1, whoami.cur); // Delete trailing comma
        whoami << "}";
    } else {
        whoami.seekp(-1, whoami.cur); // Delete trailing comma
    }

    whoami << "}";

    return whoami.str();
}

zmq::socket_t *ControllableComponent::getCtrlSocket(zmq::context_t &context,
                                                    const char *endpoint)
{
    zmq::socket_t *socket = new zmq::socket_t(context, ZMQ_DEALER);
    char id[32];
    identity(id, 32);
    socket->setsockopt(ZMQ_IDENTITY, id, std::strlen(id));
    socket->connect(endpoint);

    // Configure socket to not wait at close time
    socket->setsockopt(ZMQ_LINGER, 0);
    return socket;
}

} /* namespace oat */
