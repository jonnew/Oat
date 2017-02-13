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

#include <boost/program_options.hpp>
#include <zmq.hpp>

#include "../../lib/base/Configurable.h"
#include "../../lib/base/ControllableComponent.h"
#include "../../lib/datatypes/Frame.h"
#include "../../lib/datatypes/Pose.h"
#include "../../lib/shmemdf/Helpers.h"
#include "../../lib/shmemdf/Sink.h"
#include "../../lib/shmemdf/Source.h"
#include "../../lib/utility/Pallet.h"

namespace po = boost::program_options;

namespace oat {

static const constexpr double PI {3.141592653589793238463};

class Decorator : public ControllableComponent, public Configurable<true> {

    using pvec_size_t = oat::NamedSourceList<oat::Pose>::size_type;

public:
    /**
     * Frame decorator.
     * Adds positional, sample, and date information to frames.
     * @param frame_source_address Frame SOURCE address
     * @param frame_sink_address Decorated frame SINK address
     */
    Decorator(const std::string &frame_source_address,
              const std::string &frame_sink_address);

    // Implement ControllableComponent interface
    oat::ComponentType type(void) const override { return oat::decorator; };
    std::string name(void) const override { return name_; }

private:
    // Implement ControllableComponent interface
    virtual bool connectToNode(void) override;
    int process(void) override;
    void applyCommand(const std::string &command) override;
    oat::CommandDescription commands(void) override;

    // Implement Configurable interface
    po::options_description options() const override;
    void applyConfiguration(const po::variables_map &vm,
                            const config::OptionTable &config_table) override;

private:
    // Decorator name
    std::string name_;

    // Frame source
    std::string frame_source_address_;
    oat::Source<oat::Frame> frame_source_;

    // Frame sink for sending decorated frames
    oat::Frame shared_frame_;
    std::string frame_sink_address_;
    oat::Sink<oat::Frame> frame_sink_;

    // Poses to be added to the image stream
    oat::NamedSourceList<oat::Pose> pose_sources_;

    // Intrinsic parameters
    cv::Matx33d camera_matrix_ {cv::Matx33d::eye()};
    std::vector<double> dist_coeff_{0, 0, 0, 0, 0, 0, 0, 0};

    // Options
    bool decorate_position_{true};
    bool print_region_{false};
    bool print_timestamp_{false};
    bool print_sample_number_{false};
    bool encode_sample_number_{false};
    bool show_position_history_ {false};
    double marker_size_{1.0};
    const double symbol_alpha_{0.4};

    // Sample number encoding (automatically updated based upon frame size)
    int encode_bit_size_{5};

    // Font options
    double font_scale_{1.0};
    const int font_thickness_{1};
    const int line_thickness_{2};
    cv::Scalar font_color_{oat::RGB<oat::Roygbiv>::color(oat::Roygbiv::yellow)};
    const int font_type_{cv::FONT_HERSHEY_SIMPLEX};

    // State etc
    cv::Mat history_frame_;

    // Main frame decorate function
    void decorate(oat::Frame &frame, const std::vector<oat::Pose> &poses);

    // Frame mutating subfunctions, corresponding to different options
    void drawPose(oat::Frame &frame, const std::vector<oat::Pose> &poses);
    void printRegion(oat::Frame &frame, const std::vector<oat::Pose> &poses);
    void printTimeStamp(oat::Frame &frame);
    void printSampleNumber(oat::Frame &frame);
    void encodeSampleNumber(oat::Frame &frame);
};

}      /* namespace oat */
#endif /* OAT_DECORATOR_H */
