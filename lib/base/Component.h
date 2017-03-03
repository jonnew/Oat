//******************************************************************************
//* File:   Component.h
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

#ifndef OAT_COMPONENT_H
#define OAT_COMPONENT_H

#include <csignal>
#include <cstdlib>
#include <string>
#include <cstring>
#include <map>

#include <boost/program_options.hpp>
#include <zmq.hpp>

#include "Globals.h"

#define COMPONENT_HEARTBEAT_MS 300

namespace oat {

typedef std::map<std::string, std::string> CommandDescription;

enum ComponentType : uint16_t {
    mock = 0,
    buffer,
    calibrator,
    frameserver,
    framefilter,
    framedecorator,
    positioncombiner,
    positiondetector,
    positionfilter,
    positiongenerator,
    positionsocket,
    recorder,
    viewer,
    decorator,
    COMP_N // Number of components
};

class Component {

public:

    Component();
    virtual ~Component() { };

    /**
     * @brief Run the component's processing loop.
     */
    virtual void run();

    /**
     * @brief Human readable component name. Usually provides indication of
     * component type and IO.
     * @return Name of component
     */
    virtual std::string name(void) const = 0;

    /**
     * @brief Get an enumerated type of component.
     * @return Enumerated type of component.
     */
    virtual oat::ComponentType type(void) const = 0;

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
     * @warn This function must be thread-safe with processing thread.
     * @param command Control message
     * @return Return code. 0 = More. 1 = Quit received.
     */
    virtual void applyCommand(const std::string &command);

    /**
     * @brief Return map comtaining a runtime commands and description of
     * action on the component as implmented with the applyCommand function.
     * @return commands/description map.
     */
    virtual oat::CommandDescription commands();

    /**
     * @brief Executes component processing loop on main thread. Sets
     * process_loop_started boolean.
     */
    void runComponent(void);

    /**
     * @brief Attach components to require shared memory segments and
     * synchronization structures.
     */
    virtual bool connectToNode(void) = 0;

    /**
     * @brief Perform processing routine.
     * @return Return code. 0 = More. 1 = End of stream.
     */
    virtual int process(void) = 0;

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
#endif /* OAT_COMPONENT_H */
