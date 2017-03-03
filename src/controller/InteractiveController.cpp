//******************************************************************************
//* File:   InteractiveController.cpp
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

#include "InteractiveController.h"

#include <chrono>
#include <iostream>

#include "../../lib/utility/IOUtility.h"

namespace oat {

void InteractiveController::scanLoop()
{
    while (!break_scan_) {
        {
            std::lock_guard<std::mutex> lock(socket_mtx_);
            scan();
        } // release lock

        // This loop's period must be longer than each component heartbeat to
        // ensure there will be a message waiting to refresh our subscription
        // list with.
        std::this_thread::sleep_for(
            std::chrono::milliseconds(COMPONENT_HEARTBEAT_MS + 50));
    }
}

void InteractiveController::execute()
{
    // Start scan loop
    scan_thread_ = std::thread([this] { scanLoop(); });

usage:
    std::cout << "Provide commands as:\n"
              << "  ID/INDEX COMMAND [ARGS]\n"
              << "and then press Enter. Or,\n"
              << "- Enter 'list' to see controllable components.\n"
              << "- Enter 'quit' to quit." << std::endl;

    while (true) {

        // Get user input as line
        std::string cmd;
        std::cout << ">>> " << std::flush;
        std::getline(std::cin, cmd);

        // Check that input is still open
        if (std::cin.eof())
            break;

        // And that the command is actually present
        if (cmd.empty())
            goto usage;

        // Parse the command
        auto cmds = oat::split(cmd);

        if (cmds[0] == "quit")
            break;

        if (cmds[0] == "list") {
            std::lock_guard<std::mutex> lock(socket_mtx_);
            std::cout << list();
            continue;
        } // release lock

        if (cmds.size() < 2)
            goto usage;

        // for (auto &&s : cmds) {
        const auto id = cmds[0];
        //std::cout << id << std::endl;
        cmds.erase(cmds.begin()); // TODO: send the whole vector for
                                  // commands that require parameters

        {
            std::lock_guard<std::mutex> lock(socket_mtx_);
            if (id[0] == 'O') // TODO: is this robust?
                send(id, cmds[0]);
            else
                send(std::stoi(id), cmds[0]);
        } // release lock

        //std::cout << cmds[0] << std::endl;
    }

    break_scan_ = true;
    scan_thread_.join();
}
} /* namespace oat */
