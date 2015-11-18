//******************************************************************************
//* File:   Decorator.h
//* Author: Jon Newman <jpnewman snail mit dot edu>
//
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
//****************************************************************************

#ifndef OAT_DECORATOR_H
#define OAT_DECORATOR_H

#include <string>
#include <vector>

#include "../../lib/shmemdf/Source.h"
#include "../../lib/shmemdf/Sink.h"
#include "../../lib/shmemdf/SharedCVMat.h"
#include "../../lib/datatypes/Position2D.h"

namespace oat {

static const constexpr double PI {3.141592653589793238463};

/**
 * Frame decorator.
 * Adds positional, sample, and date information to frames.
 */
class Decorator {
public:

    using PositionSource = std::tuple< std::string,
                                       oat::Position2D,
                                       oat::Source<oat::Position2D>* >;

    /**
     * Frame decorator.
     * Adds positional, sample, and date information to frames.
     * @param position_source_addresses SOURCE addresses
     * @param frame_source_address Frame SOURCE address
     * @param frame_sink_address Decorated frame SINK address
     */
    Decorator(const std::vector<std::string> &position_source_addresses,
              const std::string &frame_source_address,
              const std::string &frame_sink_address);

    ~Decorator(void);

    /**
     * Calibrator SOURCE must be able to connect to a NODEs from
     * which to receive frames and positions.
     */
    virtual void connectToNodes(void);

    /**
     * Acquire frame and positions from all SOURCES. Decorate the frame with
     * information specified by user options. Publish decorated frame to SINK.
     * @return SOURCE end-of-stream signal. If true, this component should exit.
     */
    bool decorateFrame(void);

    //Accessors
    void set_print_region(bool value) { print_region = value; }
    void set_print_timestamp(bool value) { print_timestamp = value; }
    void set_print_sample_number(bool value) { print_sample_number = value; }
    void set_encode_sample_number(bool value) { encode_sample_number = value; }
    std::string name(void) const { return name_; }

private:

    // Decorator name
    std::string name_;

    // Internal frame copy
    cv::Mat internal_frame_;

    // Mat client object for receiving frames
    std::string frame_source_address_;
    oat::Source<SharedCVMat> frame_source_;

    // Mat server for sending decorated frames
    cv::Mat shared_frame_;
    std::string frame_sink_address_;
    oat::Sink<SharedCVMat> frame_sink_;

    // Positions to be added to the image stream
    std::vector<PositionSource> position_sources_;

    // Drawing constants
    // TODO: These may need to become a bit more sophisticated or user defined
    bool decorate_position {true};
    bool print_region {false};
    bool print_timestamp {false};
    bool print_sample_number {false};
    bool encode_sample_number {false};
    const float position_circle_radius {5.0};
    const float heading_line_length {20.0};
    const float heading_arrow_length {5};
    const float velocity_scale_factor {0.1};
    const double font_scale {1.0};
    const cv::Scalar font_color {0, 255, 0};
    const int encode_bit_size {5};

    // TODO: Look at these glorious type signatures
    void drawPosition(void);
    void drawHeading(void);
    void drawVelocity(void);
    void printRegion(void);
    void drawSymbols(void);
    void printTimeStamp(void);
    void printSampleNumber(void);
    void encodeSampleNumber(void);
};

}      /* namespace oat */
#endif /* OAT_DECORATOR_H */
