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

#pragma once

#include <csignal>
#include <cstdlib>
#include <string>
#include <cstring>

#include <boost/program_options.hpp>
#include <zmq.hpp>

#define REQUEST_RETRIES 1e6
#define REQUEST_TIMEOUT_MS 500

namespace oat {

// Global, atomic quit flag
static volatile sig_atomic_t quit {0};

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
    COMP_N // Number of components
};

class Component {

public:

    Component();
    virtual ~Component() { };

    /**
     * @brief Run the component's processing and control loops.
     */
    void run();

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
     * @brief Execultes component processing loop on main thread. Sets
     * process_loop_started boolean.
     */
    void runComponent(void);

    /**
     * @brief Start component controller on a separate thread.
     *  - Generate fresh socket basked upon socket_id
     *
     * @param ZMQ endpoint
     */
    void runController(zmq::context_t &context,
                       const char *endpoint = "ipc:///tmp/oatcomms.pipe");

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
     * @brief Attach components to require shared memory segments and
     * sychronization structures.
     */
    virtual void connectToNode(void) = 0;

    /**
     * @brief Perform processing routine.
     * @return Return code. 0 = More. 1 = End of stream.
     */
    virtual int process(void) = 0;

    /**
     * @brief Mutate component according to the requested user input. Message
     * header provides location of control struct.
     * @note This function must be thread-safe with processing thread
     * @param message_header Control message
     * @return Return code. 0 = More. 1 = Quit received.
     */
    virtual int control(const char *control_message) = 0;

private:
    zmq::socket_t *getCtrlSocket(zmq::context_t &context, const char *endpoint);
};
} /* namespace oat */
