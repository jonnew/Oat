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

#define COMPONENT_HEARTBEAT_MS 300

namespace oat {

typedef std::map<std::string, std::string> CommandDescription;

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
     * @note Only commands supplied as keys via the overridden commands()
     * function will be passed to this function.
     * @note This function must be thread-safe with processing thread.
     * @param message_header Control message
     * @return Return code. 0 = More. 1 = Quit received.
     */
    virtual void applyCommand(const std::string &command) = 0;

    /**
     * @brief Return map comtaining a runtime commands and description of
     * action on the component as implmented with the applyCommand function.
     * @return commands/description map.
     */
    virtual oat::CommandDescription commands() = 0;

private:
    /**
     * @brief Start component controller on a separate thread.
     * @param endpoint Endpoint over which communicaiton with an oat-control
     * instance will occur.
     */
    void runController(const char *endpoint = "ipc:///tmp/oatcomms.pipe");

    std::string whoAmI();

    int control(const std::string &command);
};
}      /* namespace oat */
#endif /* OAT_CONTROLLABLECOMPONENT_H */
