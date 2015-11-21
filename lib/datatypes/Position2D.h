//******************************************************************************
//* File:   Position2D.h
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

#ifndef OAT_POSITION2D_H
#define	OAT_POSITION2D_H

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include <rapidjson/prettywriter.h>

#include "Position.h"

namespace oat {

using Point2D = cv::Point2d;
using Velocity2D = cv::Point2d;
using UnitVector2D = cv::Point2d;

struct Position2D : public Position {

    Position2D(const std::string &label) :
      Position(label)
    {
        // Nothing
    }

    bool position_valid {false};
    Point2D position;

    bool velocity_valid {false};
    Velocity2D velocity;

    bool heading_valid {false};
    UnitVector2D heading;

    bool region_valid {false};
    char region[100];

    template <typename Writer>
    void Serialize(Writer& writer) const {

        writer.StartObject();

        // Sample number
        writer.String("samp");
        writer.Int(sample_);

        // Coordinate system
        writer.String("unit");
        writer.Int(static_cast<int>(unit_of_length_));

        // Position
        writer.String("pos_ok");
        writer.Bool(position_valid);

        if (position_valid) {
            writer.String("pos_xy");
            writer.StartArray();
            writer.Double(position.x);
            writer.Double(position.y);
            writer.EndArray(2);
        }

        // Velocity
        writer.String("vel_ok");
        writer.Bool(velocity_valid);

        if (velocity_valid) {
            writer.String("vel_xy");
            writer.StartArray();
            writer.Double(velocity.x);
            writer.Double(velocity.y);
            writer.EndArray(2);
        }

        // Head direction
        writer.String("head_ok");
        writer.Bool(heading_valid);

        if (heading_valid) {
            writer.String("head_xy");
            writer.StartArray();
            writer.Double(heading.x);
            writer.Double(heading.y);
            writer.EndArray(2);
        }

        // Head direction
        writer.String("reg_ok");
        writer.Bool(region_valid);

        if (region_valid) {
            writer.String("reg");
            writer.String(region);
        }

        writer.EndObject();
    }
};

}      /* namespace oat */
#endif /* OAT_POSITION2D_H */
