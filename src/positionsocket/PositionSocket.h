//******************************************************************************
//* File:   PositionSocket.h
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

#ifndef OAT_POSITIONSERVER_H
#define	OAT_POSITIONSERVER_H

#include <string>
#include <zmq.hpp>
#include <boost/asio.hpp>

#include "../../lib/datatypes/Position2D.h"
#include "../../lib/shmemdf/Source.h"
#include "../../lib/shmemdf/Sink.h"

namespace oat {

/**
 * Abstract position server.
 * All concrete position server types implement this ABC.
 */
class PositionSocket  {

public:

    PositionSocket(const std::string &position_source_address);

    virtual ~PositionSocket() { }

    /**
     * PositionSockets must be able to connect to a source
     * node in shared memory
     */
    virtual void connectToNode(void);

    /**
     * Obtain position from SOURCE. Serve position to endpoint.
     * @return SOURCE end-of-stream signal. If true, this component should exit.
     */
    bool process(void);

    // Accessors
    std::string name(void) const { return name_; }

protected:

    /**
     * Serve the position via specified IO protocol.
     * @param Position to serve.
     */
    virtual void sendPosition(const oat::Position2D &position) = 0;

private:

    // Position Socket name
    const std::string name_;

    // The position SOURCE
    std::string position_source_address_;
    oat::NodeState node_state_;
    oat::Source<oat::Position2D> position_source_;

    // The current, internally allocated position
    oat::Position2D internal_position_ {"internal"};
};

}      /* namespace oat */
#endif /* OAT_POSITIONSERVER_H */
