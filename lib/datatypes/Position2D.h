//******************************************************************************
//* File:   Position.h
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

#ifndef OAT_POSITION_H
#define	OAT_POSITION_H

#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <rapidjson/prettywriter.h>

#include "Sample.h"

namespace oat {

using Point2D = cv::Point2d;
using Velocity2D = cv::Point2d;
using UnitVector2D = cv::Point2d;

// Forward decl.
class Position2D;

/** 
 * @brief Serialize position.
 * @param p Position to serialize.
 * @param w Writer to serialize with.
 * @param verbose Allow verbose serialization.
 */
template <typename Writer>
void serializePosition(const Position2D &p, Writer &w, bool verbose = false);

/** 
 * @brief Pack a position object into a byte array.
 * @param Position2D Position to pack into a byte array.
 * @return Byte arrayByte array.
 */
std::vector<char> packPosition(const Position2D &p);

/**
 * Unit of length used to specify position.
 */
enum class DistanceUnit
{
    PIXELS = 0,   //!< Position measured in pixels. Origin is upper left.
    WORLD = 1     //!< Position measured in units specified via homography
};

class Position2D {

    template <typename Writer>
    friend void
    serializePosition(const Position2D &, Writer &, bool verbose);
    friend std::vector<char> packPosition(const Position2D &);

    using USec = Sample::Microseconds;

public:

    explicit Position2D(const std::string &label)
    {
        strncpy(label_, label.c_str(), sizeof(label_));
        label_[sizeof(label_) - 1] = '\0';
    }

    // Copy all but label, which is specific to the component
    // TODO: get rid of the label all together...
    Position2D &operator=(const Position2D &p)
    {

        // Check for self assignment
        if (this == &p)
            return *this;

        // Copy all except label_
        unit_of_length_ = p.unit_of_length_;
        sample_ = p.sample_;
        position_valid = p.position_valid;
        velocity_valid = p.velocity_valid;
        heading_valid = p.heading_valid;
        position = p.position;
        velocity = p.velocity;
        heading = p.heading;
        region_valid = p.region_valid;
        strncpy(region, p.region, sizeof(region));
        region[sizeof(region) - 1] = '\0';

        return *this;
    }

    // Accessors
    char *label() { return label_; }
    DistanceUnit unit_of_length(void) const { return unit_of_length_; }

    // Categorical position
    static constexpr size_t REGION_LEN {10};
    bool region_valid {false};
    char region[REGION_LEN] {0}; //!< Categorical position label (e.g. "North West")

    // Validity booleans
    bool position_valid {false};
    bool velocity_valid {false};
    bool heading_valid {false};

    // Position data
    Point2D position;
    Velocity2D velocity;
    UnitVector2D heading;

    // Homography
    cv::Matx33d homography() const { return homography_; }

    // Set sample rate
    void set_sample(const Sample &val) { sample_ = val; }
    void set_rate_hz(const double rate_hz) { sample_.set_rate_hz(rate_hz); }
    double sample_period_sec() const { return sample_.period_sec().count(); }
    uint64_t sample_count(void) const { return sample_.count(); }
    uint64_t sample_usec(void) const { return sample_.microseconds().count(); }
    void incrementSampleCount() { sample_.incrementCount(); }
    void incrementSampleCount(USec us) { sample_.incrementCount(us); }

    void setCoordSystem(const DistanceUnit value, const cv::Matx33d homography)
    {
        unit_of_length_ = value;
        homography_ = homography;
    }

    static constexpr size_t NPY_DTYPE_BYTES {82};
    static const char NPY_DTYPE[];

private:

    char label_[100] {0}; //!< Position label (e.g. "anterior")
    DistanceUnit unit_of_length_ {DistanceUnit::PIXELS};

    oat::Sample sample_;

    // TODO: Generalize to 3D position. Replace homography_ with tvec and rvec
    cv::Matx33d homography_ {1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0};
};

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
void serializePosition(const Position2D &p, Writer &writer, bool verbose)
{
    writer.SetMaxDecimalPlaces(5);

    writer.StartObject();

    // Sample number
    writer.String("tick");
    writer.Uint64(p.sample_count());

    writer.String("usec");
    writer.Uint64(p.sample_usec());

    // Coordinate system
    writer.String("unit");
    writer.Int(static_cast<int>(p.unit_of_length_));

    // Position
    writer.String("pos_ok");
    writer.Bool(p.position_valid || verbose);

    if (p.position_valid || verbose) {
        writer.String("pos_xy");
        writer.StartArray();
        writer.Double(p.position.x);
        writer.Double(p.position.y);
        writer.EndArray(2);
    }

    // Velocity
    writer.String("vel_ok");
    writer.Bool(p.velocity_valid || verbose);

    if (p.velocity_valid || verbose) {
        writer.String("vel_xy");
        writer.StartArray();
        writer.Double(p.velocity.x);
        writer.Double(p.velocity.y);
        writer.EndArray(2);
    }

    // Head direction
    writer.String("head_ok");
    writer.Bool(p.heading_valid);

    if (p.heading_valid || verbose) {
        writer.String("head_xy");
        writer.StartArray();
        writer.Double(p.heading.x);
        writer.Double(p.heading.y);
        writer.EndArray(2);
    }

    // Head direction
    writer.String("reg_ok");
    writer.Bool(p.region_valid);

    if (p.region_valid || verbose) {
        writer.String("reg");
        writer.String(p.region);
    }

    writer.EndObject();
}

}      /* namespace oat */
#endif /* OAT_POSITION_H */
