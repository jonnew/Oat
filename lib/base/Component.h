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

namespace oat {

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
};
}      /* namespace oat */
#endif /* OAT_COMPONENT_H */
