//******************************************************************************
//* File:   Configurable.cpp
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

#include "Configurable.h"

namespace oat {

void Configurable::appendOptions(po::options_description &opts) {

    // Default program options
    opts.add_options()
        ("config,c", po::value<std::vector<std::string>>()->multitoken(),
        "Configuration file/key pair.\n"
        "e.g. 'config.toml mykey'")
        ("control-endpoint",  po::value<std::string>(),
         "ZMQ style endpoint specifier designating runtime control port:"
         "'<transport>://<host>:<port>'. For instance, 'tcp://*:5555' to "
         "specify TCP communication on port 5555. Or, for interprocess "
         "communication: '<transport>://<user-named-pipe>. For instance "
         "'ipc:///tmp/test.pipe'. Internally, this is used to construct a "
         "ZMQ REQ socket that that receives commands from oat-control. "
         "Defaults to ipc:///tmp/oatcomms.pipe.")
        ;

    // Get type-specific options
    auto local_options = options();
    opts.add(local_options);

    // Create valid keys
    for (auto &o : local_options.options())
        config_keys_.push_back(o->long_name());
}

void Configurable::configure(const po::variables_map &vm) {

    // Check for config file and entry correctness
    auto config_table = oat::config::getConfigTable(vm);
    oat::config::checkKeys(config_keys_, config_table);

    // Concrete component uses configuration map to configure itself
    applyConfiguration(vm, config_table);
}

} /* namespace oat */
