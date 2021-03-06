//******************************************************************************
//* File:   ProgramOptions.cpp
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

#include "OatConfig.h" // Generated by CMake

#include "ProgramOptions.h"

#include <boost/program_options.hpp>
#include "make_unique.h"

namespace oat {
namespace config {

namespace po = boost::program_options;

ComponentInfo * ComponentInfo::inst = nullptr;

ComponentInfo::ComponentInfo() :
  desc(oat::make_unique<po::options_description>("INFO"))
{
    desc->add_options()
        ("help", "Produce help message.")
        ("version,v", "Print version information.")
        ;
}

ComponentInfo * ComponentInfo::instance() {

    inst = new ComponentInfo();
    return inst;
}

} /* namespace config */
} /* namespace oat */
