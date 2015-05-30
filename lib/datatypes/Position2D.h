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

namespace datatypes {

    typedef cv::Point2d Point2D;
    typedef cv::Point2d Velocity2D;
    typedef cv::Point2d UnitVector2D;

    struct Position2D : public Position {

        Position2D() : Position() { };

        // Unless manually changed, we are using pixels as our unit of measure
        int coord_system = PIXELS;

        // Used to get world coordinates from image
        bool homography_valid = false;
        cv::Matx33d homography;

        bool position_valid = false;
        Point2D position;

        bool velocity_valid = false;
        Velocity2D velocity;

        bool head_direction_valid = false;
        UnitVector2D head_direction;

        Position2D convertToWorldCoordinates() {

            if (coord_system == PIXELS && homography_valid) {
                Position2D world_position = *this;

                // Position transforms
                std::vector<Point2D> in_positions;
                std::vector<Point2D> out_positions;
                in_positions.push_back(position);
                cv::perspectiveTransform(in_positions, out_positions, homography);
                world_position.position = out_positions[0];

                // Velocity transform
                std::vector<Velocity2D> in_velocities;
                std::vector<Velocity2D> out_velocities;
                cv::Matx33d vel_homo = homography;
                vel_homo(0, 2) = 0.0; // offsets to not apply to velocity
                vel_homo(1, 2) = 0.0; // offsets to not apply to velocity
                in_velocities.push_back(velocity);
                cv::perspectiveTransform(in_velocities, out_velocities, vel_homo);
                world_position.velocity = out_velocities[0];

                // Head direction is normalized and unit-free, and therefore
                // does not require conversion

                // Return value uses world coordinates
                world_position.coord_system = WORLD;

                return world_position;

            } else {
                return *this;
            }
        }

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
            writer.String("coord");
            writer.Int(coord_system);

            // Homography (TODO: lots of overhead...)
            writer.String("hom_valid");
            writer.Bool(homography_valid);

            if (homography_valid) {
                writer.String("homography");
                writer.StartArray();
                writer.Double(homography(0,0));
                writer.Double(homography(1,0));
                writer.Double(homography(2,0));
                writer.Double(homography(0,1));
                writer.Double(homography(1,1));
                writer.Double(homography(2,1));
                writer.Double(homography(0,2));
                writer.Double(homography(1,2));
                writer.Double(homography(2,2));
                writer.EndArray(9);
            }

            // Position
            writer.String("pos_valid");
            writer.Bool(position_valid);

            if (position_valid) {
                writer.String("position");
                writer.StartArray();
                writer.Double(position.x);
                writer.Double(position.y);
                writer.EndArray(2);
            }

            // Velocity
            writer.String("vel_valid");
            writer.Bool(velocity_valid);

            if (velocity_valid) {
                writer.String("velocity");
                writer.StartArray();
                writer.Double(velocity.x);
                writer.Double(velocity.y);
                writer.EndArray(2);
            }

            // Head direction
            writer.String("headdir_valid");
            writer.Bool(head_direction_valid);

            if (head_direction_valid) {
                writer.String("headdir");
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