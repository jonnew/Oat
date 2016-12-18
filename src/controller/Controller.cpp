//******************************************************************************
//* File:   Controller.cpp
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

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include <unistd.h>

#include "Controller.h"

#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/ZMQHelpers.h"

namespace oat {

Controller::Controller(const char *endpoint)
: ctx_(1)
, router_(ctx_, ZMQ_ROUTER)
{
    router_.setsockopt(ZMQ_LINGER, 0);
    router_.bind(endpoint);
}

void Controller::send(const std::string &command)
{
    for (auto &s : subscriptions_)
        sendReqEnvelope(&router_, s.first, command);
}

void Controller::send(const std::string &command, const std::string &target_id)
{
    if (subscriptions_.count(target_id))
        sendReqEnvelope(&router_, target_id, command);
    else
        std::cerr << oat::Warn("Target is not available: " + target_id) << "\n";
}

void Controller::send(const std::string &command, const Subs::size_type idx)
{
    if (subscriptions_.size() <= idx) {
        std::cerr
            << oat::Warn("Target index out of range: " + std::to_string(idx))
            << "\n";
    } else {
        auto s = subscriptions_.begin();
        std::advance(s, idx);
        send(command, s->first);
    }
}

void Controller::scan()
{
    // Clear subscriptions in preparation for update
    subscriptions_.clear();

    // Poll the socket and find all existing connections
    int retries_left = 100;
    while (true) {

        zmq::pollitem_t p[] = {{router_, 0, ZMQ_POLLIN, 0}};
        zmq::poll(&p[0], 1, 10);

        if (p[0].revents & ZMQ_POLLIN) {

            std::string id, name;
            if (!recvReqEnvelope(&router_, id, name)) {
                std::cerr << oat::Warn("Bad receive") << "\n";
                continue;
            }

            if (addSubscriber(id, name)) {
                std::cerr << oat::Warn("Invalid component: " + id + " " + name)
                          << "\n";
                continue;
            }

        } else if (retries_left-- == 0) {
            break;
        }
    }
}

std::string Controller::list()
{
    const char sep = ' ';
    const int idx_width = 8;
    const int id_width = 22;
    const int name_width = 30;
    const int type_width = 3;
    int idx = 0;

    std::stringstream ss;
    ss << std::left << std::setw(idx_width) << std::setfill(sep) << "Index";
    ss << std::left << std::setw(id_width) << std::setfill(sep) << "ID";
    ss << std::left << std::setw(name_width) << std::setfill(sep) << "Name";
    ss << std::left << std::setw(type_width) << std::setfill(sep) << "Type";
    ss << "\n";

    for (const auto &p : subscriptions_) {

        auto sub = p.second;
        ss << std::left << std::setw(idx_width) << std::setfill(sep) << idx++;
        ss << std::left << std::setw(id_width) << std::setfill(sep) << p.first;
        ss << std::left << std::setw(name_width) << std::setfill(sep) << sub.name;
        ss << std::left << std::setw(type_width) << std::setfill(sep) << (int)sub.type;
        ss << "\n";
    }

    return ss.str();
}

int Controller::addSubscriber(const std::string &id_string,
                              const std::string &data)
{
    // Parse ID string and check for sanity
    auto header = id_string.substr(0, id_string.find("/"));
    if (header != "OAT")
        return -1;

    // Get the name and type from the data string
    auto delim_pos = data.find("/");
    if (delim_pos == std::string::npos)
        return -1;

    auto type = std::atoi(data.substr(0, delim_pos).c_str());
    auto name = data.substr(delim_pos+1);

    if (type <= oat::mock || type >= oat::COMP_N)
        return -1;

    auto ctype = static_cast<oat::ComponentType>(type);

    // Add if this component is not already
    // Check if already in list
    if (!subscriptions_.count(id_string)) {
        subscriptions_.emplace(std::piecewise_construct,
                     std::make_tuple(id_string),
                     std::make_tuple(ctype, name));
    }

    return 0;
}

} /* namespace oat */
