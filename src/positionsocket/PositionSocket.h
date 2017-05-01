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

#include <boost/program_options.hpp>

#include "../../lib/base/Component.h"
#include "../../lib/datatypes/Pose.h"
#include "../../lib/shmemdf/Sink2.h"
#include "../../lib/shmemdf/Source.h"

namespace po = boost::program_options;

namespace oat {

class PositionSocket : public Component {

public:
    /**
     * @brief An abstract Position emitter.
     * @param pose_source_addresss Position source to emit from.
     */
    explicit PositionSocket(const std::string &pose_source_addresss);
    virtual ~PositionSocket() { }

    // Component Interface
    oat::ComponentType type(void) const override { return oat::positionsocket; };

protected:
    /**
     * Send the position via specified IO protocol.
     * @param Position to serve.
     */
    virtual void sendPosition(const oat::Pose &pose) = 0;

private:
    // Component Interface
    bool connectToNode(void) override;
    int process(void) override;

    // The position SOURCE
    oat::Source<oat::Pose> pose_source_;
};

}      /* namespace oat */
#endif /* OAT_POSITIONSERVER_H */
