//******************************************************************************
//* File:   Pose.h
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

#ifndef OAT_POSE_H
#define	OAT_POSE_H

#include "Sample.h"

namespace oat {


/** 
 * @brief Generic position. Positions are defined in terms of a 3D cartesian
 * coordinate system along with the information required to map back to the
 * reference frame of the image used to deduce that position.
 */
class Pose {

public:

    using Point3D = cv::Point3d;
    using Velocity3D = cv::Point3d;

    /**
     * Unit of length used to specify position.
     */
    enum class DistanceUnit
    {
        PIXELS = 0,   //!< Position measured in pixels. Origin is upper left.
        METERS = 1    //!< Position measured in units specified via homography
    };

    //explicit Position(const std::string &label) 
    //{
    //    strncpy(label_, label.c_str(), sizeof(label_));
    //    label_[sizeof(label_) - 1] = '\0';
    //}

    //virtual ~Pose() { };

    // TODO: Why is label in here?
    //Position &operator=(const Position &p)
    //{
    //    // Check for self assignment
    //    if (this == &p)
    //        return *this;

    //    // Copy all except label_
    //    unit_of_length_ = p.unit_of_length_;
    //    sample_ = p.sample_;
    //    position_valid = p.position_valid;
    //    velocity_valid = p.velocity_valid;
    //    heading_valid = p.heading_valid;
    //    region_valid = p.region_valid;
    //    strncpy(region, p.region, sizeof(region));
    //    region[sizeof(region) - 1] = '\0';

    //    return *this;
    //}

    // What is the point of making these member's private?
    // Expose sample information
    //oat::Sample & sample() { return sample_; };
    //char * label() {return label_; }
    
    // Accessors
    DistanceUnit unit_of_length(void) const { return unit_of_length_; }
    
    // Categorical position
    bool region_valid {false};
    char region[100] {0}; //!< Categorical position label (e.g. "North West")

    // Cartesian position
    bool position_valid {false};
    Point3D position;

    // Velocity
    bool velocity_valid {false};
    Velocity3D velocity;

    // Heading (combine with position to get full pose)
    bool heading_valid {false}; 

    /** 
     * @brief Rotation vector used for mapping the position, with respect to
     * its coordinate frame, back to pixels with respect to the camera's
     * coordinate frame. Apply transform using `cv::Affine3f`.
     */
    cv::Matx13d rvec {0, 0, 0};
    //double rvec[3] {0, 0, 0};
    //std::vector<double> rvec {0, 0, 0};

    /** 
     * @brief Translation vector used for mapping the position, with respect to
     * its coordinate frame, back to pixels with respect to the camera's
     * coordinate frame. Apply transform using `cv::Affine3f`.
     */
    cv::Matx13d tvec {0, 0, 0};
    //double tvec[3] {0, 0, 0};
    //std::vector<double> tvec {0, 0, 0};

    /** 
     * @brief Camera matrix
     */
    //cv::Matx33d camera {1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0};

    /** 
     * @brief Camera lens distortion coefficients       
     */
    //std::vector<double> distortion_coefficients {0, 0, 0, 0, 0};

    /**
     * @brief JSON Serializer
     *
     * @param writer Writer to use for serialization
     * @param verbose If true, specifies that fields be serialized even though
     * they contain indeterminate data? This is useful for ease of sample
     * alignment during post processing of saved files.
     */
    // TODO: Should this just return a rapidjson::Document which is redily
    // transformed into other formats such as msgpack, etc, instead of doing the 
    // writing right here?
    template <typename Writer>
    void Serialize(Writer &writer, bool verbose = false) const
    {
        // 0.001 pixels or 1 mm precision
        writer.SetMaxDecimalPlaces(3);

        writer.StartObject();

        // Sample number
        writer.String("tick");
        writer.Uint64(sample_.count());

        writer.String("usec");
        writer.Uint64(sample_.microseconds().count());

        // Coordinate system
        writer.String("unit");
        writer.Int(static_cast<int>(unit_of_length_));

        // Position
        writer.String("pos_ok");
        writer.Bool(position_valid || verbose);

        if (position_valid || verbose) {
            writer.String("pos");
            writer.StartArray();
            writer.Double(position.x);
            writer.Double(position.y);
            writer.Double(position.z);
            writer.EndArray(3);
        }

        // Velocity
        writer.String("vel_ok");
        writer.Bool(velocity_valid || verbose);

        if (velocity_valid || verbose) {
            writer.String("vel");
            writer.StartArray();
            writer.Double(velocity.x);
            writer.Double(velocity.y);
            writer.Double(velocity.z);
            writer.EndArray(3);
        }

        // Pose
        writer.String("head_ok");
        writer.Bool(heading_valid);

        if (heading_valid || verbose) {
            writer.String("rvec");
            writer.StartArray();
            writer.Double(rvec(0));
            writer.Double(rvec(1));
            writer.Double(rvec(2));
            writer.EndArray(3);

            writer.String("tvec");
            writer.StartArray();
            writer.Double(tvec(0));
            writer.Double(tvec(1));
            writer.Double(tvec(2));
            writer.EndArray(3);
        }

        // Region
        writer.String("reg_ok");
        writer.Bool(region_valid);

        if (region_valid || verbose) {
            writer.String("reg");
            writer.String(region);
        }

        writer.EndObject();
    }

    void setCoordSystem(const DistanceUnit unit, const cv::Matx33d h)
    {
        unit_of_length_ = unit;
        homography_ = h;
    }

protected:
   
    // TODO: What is the label even used for? For decoration, but really it
    // just refects to source of the signal, which we should be able to get
    // from the source anyway
    //char label_[100] {0}; //!< Position label (e.g. "anterior")
    DistanceUnit unit_of_length_ {DistanceUnit::PIXELS};
    
    /** 
     * @brief Timing information
     */
    oat::Sample sample_;

    /** 
     * @brief Homography
     */
    cv::Matx33d homography_ {1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0};
};

}      /* namespace oat */
#endif /* OAT_POSE_H */

