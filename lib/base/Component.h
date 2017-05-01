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

#include "../utility/TOMLSanitize.h"

#include "Globals.h"

#define COMPONENT_HEARTBEAT_MS 300

namespace oat {

namespace po = boost::program_options;
using CommandDescription =  std::map<std::string, std::string>;

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

const std::map<ComponentType, std::string> ComponentString = {
    {mock ,                "mock"},
    {buffer,               "buffer"},
    {calibrator,           "calibrate"},
    {frameserver,          "frameserve"},
    {framefilter,          "framefilt"},
    {framedecorator,       "decorate"},
    {positioncombiner,     "posecom"},
    {positiondetector,     "posedet"},
    {positionfilter,       "posefilt"},
    {positiongenerator,    "posegen"},
    {positionsocket,       "posesock"},
    {recorder,             "recorder"},
    {viewer,               "view"}
};

class Component {

public:

    Component();
    Component(const std::string &source_addr, const std::string &sink_addr);
    Component(const std::vector<std::string> &source_addr, const std::string &sink_addr);
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
    std::string name(void) const { return name_; }

    /**
     * @brief Get an enumerated type of component.
     * @return Enumerated type of component.
     */
    virtual oat::ComponentType type(void) const = 0;

    /**
     * @brief Append command line program options.
     * @param opts Program option description to be specialized.
     */
    void appendOptions(po::options_description &opts);

    /**
     * @brief Configure program runtime parameters.
     * @param vm Previously parsed program option value map.
     */
    void configure(const po::variables_map &vm);

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

    /**
     * @brief Return the component's program options.
     * @return Program options specilized for a particular concrete component
     * type.
     */
    virtual po::options_description options(void) const = 0;

    /**
     * @brief Apply type-specific component configurations using a pre-parsed program option map.
     * @param vm Pre-parse program option map.
     * @param config_table Parsed TOML options table.
     */
    virtual void applyConfiguration(const po::variables_map &vm,
                                    const config::OptionTable &config_table) = 0;

    void set_name(const std::string &source_addr, const std::string &sink_addr);

    void set_name(const std::vector<std::string> &source_addrs,
                  const std::string &sink_addr);

private:
    /**
     * @brief Start component controller on a separate thread.
     * @param endpoint Endpoint over which communicaiton with an oat-control
     * instance will occur.
     */
    void runController(const char *endpoint = "ipc:///tmp/oatcomms.pipe");

    /**
     * @brief Produce complete component description.
     * @return JSON string with with component name, type, unique ID, and
     * command hash.
     */
    std::string whoAmI();

    /**
     * @brief Preprocess received control action (e.g. figure out if we need to
     * quit) and then forward the command to component(s) if appropriate.
     * @param command Command to send.
     * @return 1 if quit command received, 0 otherwise.
     */
    int control(const std::string &command);

    /**
     * @brief Allowable program option keys
     */
    std::vector<std::string> config_keys_;

    /** 
     * @brief Human readable component name
     */
    std::string name_{""};
};

}      /* namespace oat */
#endif /* OAT_COMPONENT_H */
