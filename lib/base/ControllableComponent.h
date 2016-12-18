//******************************************************************************
//* File:   ControllableComponent.h
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

#ifndef OAT_CONTROLLABLECOMPONENT_H
#define OAT_CONTROLLABLECOMPONENT_H

#include <csignal>
#include <cstdlib>
#include <string>
#include <cstring>
#include <map>

#include <boost/program_options.hpp>
#include <zmq.hpp>

#include "Component.h"
#include "Globals.h"

#define REQUEST_RETRIES 1e6
#define REQUEST_TIMEOUT_MS 500
#define OAT_CONTROLLABLE

namespace oat {

using CommandHash = std::map<std::string, int>;

class ControllableComponent : public Component {

public:

    using Component::Component;
    virtual ~ControllableComponent() { };

    /**
     * @brief Run the component's processing and control loops.
     */
    void run() override;

protected:

    /**
     * @brief Start component controller on a separate thread.
     *  - Generate fresh socket basked upon socket_id
     *
     * @param ZMQ endpoint
     */
    void runController(const char *endpoint = "ipc:///tmp/oatcomms.pipe");

    /**
     * @brief Get unique, controllable ID for this component
     * @param n Number of characters to copy to id
     * @param ASCII string ID consisting of the character 'C' followed by a
     * serilized ComponentType component enumerator, then a '.' delimeter, and
     * then seralized string representing the handle of the component control
     * thread.
     */
    void identity(char *id, const size_t n) const;

    /**
     * @brief Mutate component according to the requested user input. Message
     * header provides location of control struct.
     * @note This function must be thread-safe with processing thread
     * @param message_header Control message
     * @return Return code. 0 = More. 1 = Quit received.
     */
    virtual void applyCommand(const std::string &command) = 0;

private:

    int control(const std::string &command);

    zmq::socket_t *getCtrlSocket(zmq::context_t &context, const char *endpoint);
};
}      /* namespace oat */
#endif /* OAT_CONTROLLABLECOMPONENT_H */
