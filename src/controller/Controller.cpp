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

#include "Controller.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include <unistd.h>

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
    for (const auto &s : subscriptions_)
        sendReqEnvelope(router_, s.first, command);
}

void Controller::send(const std::string &target_id, const std::string &command)
{
    if (subscriptions_.count(target_id)) {
        if (command == "help" || command == "Help")
            help(target_id);
        else
            sendReqEnvelope(router_, target_id, command);
    } else {
        std::cerr << oat::Warn("Target is not available: " + target_id) << "\n";
    }
}

void Controller::send(const Subs::size_type idx, const std::string &command)
{
    if (subscriptions_.size() <= idx) {
        std::cerr
            << oat::Warn("Target index out of range: " + std::to_string(idx))
            << "\n";
    } else {
        auto s = subscriptions_.begin();
        std::advance(s, idx);
        send(s->first, command);
    }
}

void Controller::scan()
{
    // Clear subscriptions in preparation for update
    subscriptions_.clear();

    // Poll the socket and find all existing connections
    bool more;
    do {

        zmq::pollitem_t p[] = {{router_, 0, ZMQ_POLLIN, 0}};
        zmq::poll(&p[0], 1, 10);

        more = (p[0].revents & ZMQ_POLLIN);
        if (more) {

            std::string id, info;
            if (!recvReqEnvelope(router_, id, info)) {
                std::cerr << oat::Warn("Bad receive") << "\n";
                continue;
            }

            if (addSubscriber(id, info)) {
                std::cerr << oat::Warn("Invalid component: " + id + " " + info)
                          << "\n";
                continue;
            }
        } 
    } while (more);
}

std::string Controller::list() const
{
    const char sep = ' ';
    const int idx_width = 8;
    const int id_width = 22;
    const int name_width = 30;
    const int type_width = 3;
    int idx = 0;

    std::stringstream ss;

    if (subscriptions_.size() == 0) 
        return ss.str();

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

    // Parse the JSON string containing type, name, and usage info
    rapidjson::Document sub_info;
    sub_info.Parse(data.c_str());
    assert(sub_info.IsObject());

    // Get subscriber type
    assert(sub_info.HasMember("type"));
    assert(sub_info["type"].IsInt());
    assert(sub_info["type"].GetInt() < oat::COMP_N);
    assert(sub_info["type"].GetInt() >= oat::mock);
    auto ctype = static_cast<oat::ComponentType>(sub_info["type"].GetInt());

    // Get subscriber type
    assert(sub_info.HasMember("name"));
    assert(sub_info["name"].IsString());
    auto name = std::string(sub_info["name"].GetString());

    // Get description and format if any are provided
    oat::CommandDescription desc_map;

    if (sub_info.HasMember("commands")) {
        const rapidjson::Value &desc = sub_info["commands"];
        assert(desc.IsObject());

        for (auto &&d : desc.GetObject()) {
            assert(d.value.IsString());
            desc_map.emplace(d.name.GetString(), d.value.GetString());
        }
    }

    // Add if this component is not already in hash
    if (!subscriptions_.count(id_string)) {
        subscriptions_.emplace(std::piecewise_construct,
                               std::make_tuple(id_string),
                               std::make_tuple(ctype, name, desc_map));
    } 

    return 0;
}

void Controller::help(const std::string &target_id) const
{
    if (subscriptions_.count(target_id)) {

        auto cmds = subscriptions_.at(target_id).commands;
        cmds.emplace("help", "Print this message.");
        cmds.emplace("quit", "Exit the program.");

        size_t max_len = 7; // For "COMMAND"

        for (const auto &c : cmds) {
            if (c.first.size() > max_len)
                max_len = c.first.size();
        }
        max_len += 5;

        // Format help message
        std::stringstream out;
        out << std::left << subscriptions_.at(target_id).name << "\n";
        out << std::left << std::setw(max_len) << "COMMAND " << "DESC.\n";

        int desc_len = 80 - max_len;
        assert(desc_len > 0);
        for (auto &&c : cmds) {
            out <<  std::left << std::setw(max_len) << (" " + c.first);

            size_t from = 0;
            size_t to = 0;
            while (to < c.second.size()) {
                to = c.second.find(" ", to + desc_len);
                if (from != 0)
                    out << std::left << std::setw(max_len - 1) << "";
                out << c.second.substr(from, to) << "\n";

                from = to;
            }
        }

        std::cout << out.str();

    } else {
        std::cerr << oat::Warn("Target is not available: " + target_id) << "\n";
    }
}

} /* namespace oat */
