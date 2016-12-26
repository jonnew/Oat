//******************************************************************************
//* File:   PositionCout.h
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

#ifndef OAT_POSITIONCOUT_H
#define	OAT_POSITIONCOUT_H

#include "PositionSocket.h"

#include <string>

namespace oat {

// Forward decl.
class Position2D;

class PositionCout : public PositionSocket {

    // Use base constructor
    using PositionSocket::PositionSocket;

    // Configurable Interface
    po::options_description options() const override;
    void applyConfiguration(const po::variables_map &vm,
                            const config::OptionTable &config_table) override;

    // Format std out stream
    bool pretty_ {false};

    void sendPosition(const oat::Position2D &position) override;
};

}      /* namespace oat */
#endif /* OAT_POSITIONCOUT_H */
