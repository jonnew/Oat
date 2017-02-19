//******************************************************************************
//* File:   Controller.h
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

#ifndef OAT_CONTROLLER_H
#define OAT_CONTROLLER_H

#include <string>
#include <unordered_map>

#include "rapidjson/document.h"
#include "zmq.hpp"

#include "../../lib/base/ControllableComponent.h"

namespace oat {

class Controller {

    struct Subscriber {
        Subscriber(const oat::ComponentType type,
                   const std::string &name,
                   const oat::CommandDescription &commands)
        : type(type)
        , name(name)
        , commands(commands) { }

        const oat::ComponentType type;
        const std::string name;
        const oat::CommandDescription commands;
    };

public:
    using Identity = std::string;
    using Subs = std::map<Identity, Subscriber>;

    explicit Controller(const char *endpoint);

    /**
     * @brief Find out which clients are available on the socket. Update
     * subscriptions_ hash.
     */
    void scan();

    void send(const std::string &target_id, const std::string &cmd);
    void send(const Subs::size_type idx, const std::string &cmd);
    void send(const std::string &cmd);

    std::string list(void) const;

    int addSubscriber(const std::string &identity,
                      const std::string &name);

protected:

    // Hashed subscriptions
    Subs subscriptions_;

private:

    void help(const std::string &target_id) const;
    //void printHelp(const oat::CommandDescription &cmds) const;

    // Router socket
    zmq::context_t ctx_;
    zmq::socket_t router_;
};

}      /* namespace oat */
#endif /* OAT_CONTROLLER_H */
