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

#include "../../lib/datatypes/Frame.h"
#include "../../lib/shmemdf/Helpers.h"
#include "../../lib/shmemdf/Source.h"
#include "../../lib/shmemdf/Sink.h"
#include "../../lib/shmemdf/SharedFrameHeader.h"
#include "../../lib/datatypes/Position2D.h"

namespace oat {

static const constexpr double PI {3.141592653589793238463};

/**
 * Frame decorator.
 * Adds positional, sample, and date information to frames.
 */
class Decorator {

    using pvec_size_t = oat::NamedSourceList<oat::Position2D>::size_type;

public:

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

    /**
     * Calibrator SOURCE must be able to connect to a NODEs from
     * which to receive frames and positions.
     */
    void connectToNodes(void);

    /**
     * Acquire frame and positions from all SOURCES. Decorate the frame with
     * information specified by user options. Publish decorated frame to SINK.
     * @return SOURCE end-of-stream signal. If true, this component should exit.
     */
    bool decorateFrame(void);

    //Accessors
    void set_print_region(bool value) { print_region_ = value; }
    void set_print_timestamp(bool value) { print_timestamp_ = value; }
    void set_print_sample_number(bool value) { print_sample_number_ = value; }
    void set_encode_sample_number(bool value) { encode_sample_number_ = value; }
    void set_show_position_history(bool value) { show_position_history_ = value; }

    std::string name(void) const { return name_; }

private:

    // Decorator name
    std::string name_;

    // Internal frame copy
    oat::Frame internal_frame_;

    // Mat client object for receiving frames
    std::string frame_source_address_;
    oat::Source<SharedFrameHeader> frame_source_;

    // Mat server for sending decorated frames
    oat::Frame shared_frame_;
    std::string frame_sink_address_;
    oat::Sink<SharedFrameHeader> frame_sink_;

    // Positions to be added to the image stream
    std::vector<oat::Position2D> positions_;
    oat::NamedSourceList<oat::Position2D> position_sources_;

    // Options
    bool decorate_position_ {true};
    bool print_region_ {false};
    bool print_timestamp_ {false};
    bool print_sample_number_ {false};
    bool encode_sample_number_ {false};

    // TODO: These may need to become a bit more sophisticated or user defined
    // Position drawing
    const double symbol_scale_ {0.01};
    const double velocity_scale_factor_ {0.15};
    double position_circle_radius_ {2.0};
    double heading_line_length_ {8.0};
    bool show_position_history_ {false};
    std::vector<bool> positions_found_;
    std::vector<oat::Point2D> previous_positions_;
    cv::Mat history_frame_;
    const double symbol_alpha_ {0.4};
    const cv::Scalar pos_colors_[12] {CV_RGB(255,  51,  51),
                                      CV_RGB( 51, 255,  51),
                                      CV_RGB( 51,  51, 255),
                                      CV_RGB(255, 153,  51),
                                      CV_RGB( 51, 255, 153),
                                      CV_RGB(255,  51, 153),
                                      CV_RGB(255, 255,  51),
                                      CV_RGB( 51, 255, 255),
                                      CV_RGB(255,  51, 255),
                                      CV_RGB(153, 255,  51),
                                      CV_RGB( 51, 153, 255),
                                      CV_RGB(153,  51, 255)};

    // Font
    const double font_scale_ {1.0};
    const int font_thickness_ {1};
    const int line_thickness_ {2};
    const cv::Scalar font_color_ {255, 255, 255};
    const int font_type_ {cv::FONT_HERSHEY_SIMPLEX};

    // Sample number encoding
    int encode_bit_size_ {5};

    /**
     * Project Positions into oat::PIXEL coordinates.
     * @param pos Position with unit_of_length != oat::PIXEL to be converted to
     * unit_of_length == oat::PIXEL.
     */
    void invertHomography(oat::Position2D &pos);

    // TODO: Look at these glorious type signatures. These are 'subroutines'
    // rather than functions...
    void drawPosition(void);
    void printRegion(void);
    void drawOnFrame(void);
    void printTimeStamp(void);
    void printSampleNumber(void);
    void encodeSampleNumber(void);
};

}      /* namespace oat */
#endif /* OAT_DECORATOR_H */
