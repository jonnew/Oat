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

#ifndef OAT_CONFIGURABLE_H
#define OAT_CONFIGURABLE_H

#include <string>
#include <unordered_map>

#include "zmq.hpp"

#include "../../lib/base/Component.h"

namespace oat {

struct Subscriber {
    Subscriber(const oat::ComponentType type, const std::string &name)
    : type(type)
    , name(name) { }

    oat::ComponentType type;
    std::string name;
};

class Controller {

public:

    using Identity = std::string;
    using Subs = std::map<Identity, oat::Subscriber>;

    Controller(const char *endpoint);

    /**
     * @brief Find out which clients are available on the socket. Update
     * subscriptions_ hash.
     */
    void scan();

    void send(const std::string &cmd, const std::string &target_id);
    void send(const std::string &cmd, const Subs::size_type idx);
    void send(const std::string &cmd);

    std::string list(void);

    int addSubscriber(const std::string &identity,
                      const std::string &name);

private:

    // Hashed subscriptions 
    Subs subscriptions_;

    // Router socket
    zmq::context_t ctx_;
    zmq::socket_t router_;
};
}      /* namespace oat */
#endif /* OAT_CONTROLLER_H */
