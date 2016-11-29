//******************************************************************************
//* File:   Configurable.h
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

namespace oat {

namespace po = boost::program_options;

class Configurable {

public:

    /**
     * @brief Append program options.
     * @param opts Program option description to be specialized.
     */
    void appendOptions(po::options_description &opts);

    /**
     * @brief Configure program parameters.
     * @param vm Previously parsed program option value map.
     */
    void configure(const po::variables_map &vm);

protected:

    /**
     * @brief Return the component's program options.
     * @return Program options specilized for a particular concrete component
     * type.
     */
    virtual po::options_description options(void) const = 0;

    /**
     * @brief Apply type-specific component configurations using a pre-parsed program option map.
     * @param vm Pre-parse program option map.
     */
    virtual void applyConfiguration(const po::variables_map &vm) = 0;

    std::vector<std::string> config_keys_;
};
} /* namespace oat */
