//******************************************************************************
//* Copyright (c) Jon Newman (jpnewman at mit snail edu) 
//* All right reserved.
//* This file is part of the Simple Tracker project.
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

#ifndef POSITION2D_H
#define	POSITION2D_H

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include "../../lib/rapidjson/prettywriter.h"

#include "Position.h"

namespace oat {

    typedef cv::Point2d Point2D;
    typedef cv::Point2d Velocity2D;
    typedef cv::Point2d UnitVector2D;

    struct Position2D : public Position {

        Position2D() : Position() { };

        bool position_valid = false;
        Point2D position;

        bool velocity_valid = false;
        Velocity2D velocity;

        bool head_direction_valid = false;
        UnitVector2D head_direction;

        template <typename Writer>
        void Serialize(Writer& writer, const std::string& label) const {
            
            writer.StartObject();

            // Name
            writer.String("ID");
#ifdef RAPIDJSON_HAS_STDSTRING
            writer.String(label);
#else
            writer.String(label.c_str(), (rapidjson::SizeType)label.length()); // Supplying length of string is faster.
#endif

            // Coordinate system
            writer.String("unit");
            writer.Int(coord_system);

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
            writer.Bool(head_direction_valid);

            if (head_direction_valid) {
                writer.String("head_xy");
                writer.StartArray();
                writer.Double(head_direction.x);
                writer.Double(head_direction.y);
                writer.EndArray(2);
            }

            writer.EndObject();

        }
    };
} // namespace datatypes

#endif	/* POSITION2D_H */