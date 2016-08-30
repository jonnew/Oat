//******************************************************************************
//* File:   Decorator.cpp
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

#include <cmath>
#include <exception>
#include <string>
#include <tuple>
#include <vector>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <cmath>

#include "../../lib/datatypes/Position2D.h"
#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/TOMLSanitize.h"
#include "../../lib/utility/make_unique.h"

#include "Decorator.h"

namespace oat {

Decorator::Decorator(const std::string &frame_source_address,
                     const std::string &frame_sink_address) :
  name_("decorator[" + frame_source_address+ "->" + frame_sink_address + "]")
, frame_source_address_(frame_source_address)
, frame_sink_address_(frame_sink_address)
{
    // Nothing
}

void Decorator::appendOptions(po::options_description &opts) {

    opts.add_options()
        ("config,c", po::value<std::vector<std::string> >()->multitoken(),
        "Configuration file/key pair.\n"
        "e.g. 'config.toml mykey'")
        ;

    // Update CLI options
    po::options_description local_opts;
    local_opts.add_options()
        ("position-sources,p", po::value< std::vector<std::string> >()->multitoken(),
        "The name of position SOURCE(s) used to draw object position markers.\n")
        ("timestamp,t", "Write the current date and time on each frame.\n")
        ("sample,s", "Write the frame sample number on each frame.\n")
        ("sample-code,S", "Write the binary encoded sample on the corner of each frame.\n")
        ("region,R", "Write region information on each frame "
        "if there is a position stream that contains it.\n")
        ("history,h", "Display position history.\n")
        ;

    opts.add(local_opts);

    // Return valid keys
    for (auto &o: local_opts.options())
        config_keys_.push_back(o->long_name());
}

void Decorator::configure(const po::variables_map &vm) {

    // Check for config file and entry correctness
    auto config_table = oat::config::getConfigTable(vm);
    oat::config::checkKeys(config_keys_, config_table);

    // Position sources
    // NOTE: not setable via configuration file
    if (vm.count("position-sources"))  {

        auto p_source_addrs = 
            vm["position-sources"].as< std::vector<std::string> >();

        // Setup position sources
        if (!p_source_addrs.empty()) {
            for (auto &addr : p_source_addrs) {

                oat::Position2D pos(addr);
                positions_.push_back(std::move(pos));
                position_sources_.push_back(
                    oat::NamedSource<oat::Position2D>(
                        addr,
                        std::make_unique<oat::Source<oat::Position2D>>()
                    )
                );
            }
        } else {
            decorate_position_ = false;
        }
    }

    // Timestamp
    oat::config::getValue<bool>(vm, config_table, "timestamp", print_timestamp_);

    // Region
    oat::config::getValue<bool>(vm, config_table, "region", print_region_);

    // Sample number
    oat::config::getValue<bool>(vm, config_table, "sample", print_sample_number_);

    // Sample number
    oat::config::getValue<bool>(vm, config_table, "sample-code", encode_sample_number_);

    // Path history
    oat::config::getValue<bool>(vm, config_table, "history", show_position_history_);
}

void Decorator::connectToNodes() {

    // Examine sample period of sources to make sure they are the same
    double sample_rate_hz;
    std::vector<double> all_ts;

    // Establish our a slots in the frame and positions sources
    frame_source_.touch(frame_source_address_);

    for (auto &ps : position_sources_)
        ps.source->touch(ps.name);

    // Wait for synchronous start with sink when it binds the node
    frame_source_.connect();

    for (auto &ps : position_sources_) {
        ps.source->connect();
        all_ts.push_back(ps.source->retrieve()->sample().period_sec().count());
    }

    // Get frame meta data to format sink
    oat::Source<oat::Frame>::ConnectionParameters param =
            frame_source_.parameters();

    // Bind to sink sink node and create a shared frame
    frame_sink_.bind(frame_sink_address_, param.bytes);
    shared_frame_ = frame_sink_.retrieve(param.rows, param.cols, param.type);
    all_ts.push_back(shared_frame_.sample().period_sec().count());

    if (!oat::checkSamplePeriods(all_ts, sample_rate_hz)) {
        std::cerr << oat::Warn(oat::inconsistentSampleRateWarning(sample_rate_hz));
    }

    // Set drawing parameters based on frame dimensions
    const size_t min_size = (param.rows < param.cols) ? param.rows : param.cols;
    position_circle_radius_ = std::ceil(symbol_scale_ * min_size);
    heading_line_length_ = std::ceil(symbol_scale_ * min_size);
    encode_bit_size_  =
        std::ceil(param.cols / 3 / sizeof(internal_frame_.sample().count()) / 8);

    // If we are drawing positions, get ready for that
    if (decorate_position_) {
        previous_positions_.push_back(oat::Point2D(0,0));
        positions_found_.push_back(false);
        history_frame_ = cv::Mat::zeros(shared_frame_.size(), shared_frame_.type());
    }
}

bool Decorator::process() {

    // 1. Get frame
    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sink to write to node
    if (frame_source_.wait() == oat::NodeState::END)
        return true;

    // Clone the shared frame
    frame_source_.copyTo(internal_frame_);

    // Tell sink it can continue
    frame_source_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // 2. Get positions
    for (pvec_size_t i = 0; i !=  position_sources_.size(); i++) {

        // START CRITICAL SECTION //
        ////////////////////////////
        if (position_sources_[i].source->wait() == oat::NodeState::END)
            return true;

        positions_[i] = position_sources_[i].source->clone();

        position_sources_[i].source->post();
        ////////////////////////////
        //  END CRITICAL SECTION  //
    }

    // Decorate frame
    drawOnFrame();

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sources to read
    frame_sink_.wait();

    internal_frame_.copyTo(shared_frame_);

    // Tell sources there is new data
    frame_sink_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // None of the sink's were at the END state
    return false;
}

void Decorator::drawOnFrame() {

    if (decorate_position_) {

        drawPosition();

        if (print_region_)
            printRegion();
    }

    if (print_timestamp_)
        printTimeStamp();

    if (print_sample_number_)
        printSampleNumber();

    if (encode_sample_number_)
        encodeSampleNumber();
}

void Decorator::invertHomography(oat::Position2D &p) {

    if (p.position_valid) {

        cv::Matx33d inv_homo = p.homography().inv();

        std::vector<oat::Point2D> in_positions;
        std::vector<oat::Point2D> out_positions;
        in_positions.push_back(p.position);
        cv::perspectiveTransform(in_positions, out_positions, inv_homo);
        p.position = out_positions[0];

        if (p.velocity_valid) {

            std::vector<oat::Velocity2D> in_velocities;
            std::vector<oat::Velocity2D> out_velocities;
            cv::Matx33d vel_inv_homo = inv_homo;
            vel_inv_homo(0, 2) = 0.0; // offsets do not apply to velocity
            vel_inv_homo(1, 2) = 0.0; // offsets do not apply to velocity
            in_velocities.push_back(p.velocity);
            cv::perspectiveTransform(in_velocities, out_velocities, vel_inv_homo);
            p.velocity = out_velocities[0];
        }

        if (p.heading_valid) {

            std::vector<oat::UnitVector2D> in_heading;
            std::vector<oat::UnitVector2D> out_heading;
            cv::Matx33d head_inv_homo = inv_homo;
            head_inv_homo(0, 2) = 0.0; // offsets do not apply to heading
            head_inv_homo(1, 2) = 0.0; // offsets do not apply to heading
            in_heading.push_back(p.heading);
            cv::perspectiveTransform(in_heading, out_heading, head_inv_homo);
            cv::normalize(out_heading, out_heading);
            p.heading = out_heading[0];
        }
    }
}

void Decorator::drawPosition() {

    size_t i = 0;

    cv::Mat symbol_frame =
        cv::Mat::zeros(internal_frame_.size(), internal_frame_.type());

    for (auto &p : positions_) {

        if (p.unit_of_length() == oat::DistanceUnit::WORLD)
            invertHomography(p);

        if (p.position_valid) {

            cv::circle(symbol_frame,
                       p.position,
                       position_circle_radius_,
                       pos_colors_[i],
                       line_thickness_);

            if (show_position_history_ && positions_found_[i]) {

                cv::line(history_frame_,
                         p.position,
                         previous_positions_[i],
                         pos_colors_[i],1);
            }

            previous_positions_[i] = p.position;

            if (p.velocity_valid) {

                cv::Point2d end =
                    p.position + (velocity_scale_factor_ * p.velocity);
                cv::line(symbol_frame,
                         p.position,
                         end,
                         pos_colors_[i],
                         line_thickness_);
            }

            if (p.heading_valid) {

                cv::Point2d start =
                    p.position - (heading_line_length_ * p.heading);
                cv::Point2d end =
                    p.position + (1.5 * heading_line_length_ * p.heading);

                cv::arrowedLine(symbol_frame,
                                start,
                                end,
                                font_color_,
                                line_thickness_);
            }

            positions_found_[i] = true;
        } else {
            positions_found_[i] = false;
        }

        (i > position_sources_.size() - 1) ? i = 0 : i++;
    }

    // TODO: Following is inefficient and it shows in the performance testing
    if (show_position_history_)
        symbol_frame += history_frame_;

    cv::Mat result_frame =
        cv::Mat::zeros(internal_frame_.size(), internal_frame_.type());
    cv::addWeighted(internal_frame_,
                    1 - symbol_alpha_,
                    symbol_frame,
                    symbol_alpha_,
                    0.0,
                    result_frame);

    cv::Mat mask;
    const cv::Scalar zero(0);
    cv::inRange(symbol_frame, zero, zero, mask);
    internal_frame_.setTo(zero, mask == 0);
    result_frame.setTo(zero, mask);
    internal_frame_ += result_frame;
}


void Decorator::printRegion() {

    // Create display string
    std::string reg_text;

    if (position_sources_.size() == 1)
        reg_text = "Region:";
    else
        reg_text = "Regions:";

    // Calculate text origin based upon message size
    int baseline = 0;
    cv::Size reg_text_size =
            cv::getTextSize(reg_text, font_type_, font_scale_, font_thickness_, &baseline);

    cv::Point text_origin(10, reg_text_size.height);
    cv::putText(internal_frame_, reg_text, text_origin, font_thickness_, font_scale_, font_color_);

    // Add ID: region information
    size_t i = 0;
    for (auto &ps : position_sources_) {
        if (positions_[i].region_valid)
            reg_text = ps.name + ": " + std::string(positions_[i].region);
        else
            reg_text = ps.name + ": ?";

        text_origin.y += reg_text_size.height + 2;
        cv::putText(internal_frame_,
                    reg_text, text_origin,
                    font_thickness_,
                    font_scale_,
                    pos_colors_[i]);

        (i > position_sources_.size() - 1) ? i = 0 : i++;
    }
}

void Decorator::printTimeStamp() {

    std::time_t raw_time;
    struct tm * time_info;
    char buffer[80];

    std::time(&raw_time);
    time_info = std::localtime(&raw_time);

    std::strftime(buffer, 80, "%c", time_info);

    cv::Point text_origin(internal_frame_.cols - 230, internal_frame_.rows - 10);
    cv::putText(internal_frame_, std::string(buffer), text_origin, 1, font_scale_, font_color_);
}

void Decorator::printSampleNumber() {

    cv::Point text_origin(10, internal_frame_.rows - 10);
    cv::putText(internal_frame_,
                std::to_string(internal_frame_.sample().count()),
                text_origin,
                1,
                font_scale_,
                font_color_);
}

void Decorator::encodeSampleNumber() {

    uint64_t sample_count = internal_frame_.sample().count();
    int column = internal_frame_.cols - 64 * encode_bit_size_;

    if (column < 0)
        throw std::runtime_error("Binary counter bar is too large for frame."
                                 "Use more x-dim pixels or turn binary counter off.");

    for (int shift = 0; shift < 64; shift++) {

        cv::Mat sub_square = internal_frame_.colRange(column, column + encode_bit_size_).rowRange(0, encode_bit_size_);

        if (sample_count & 0x1) {

            cv::Mat true_mat(encode_bit_size_, encode_bit_size_, internal_frame_.type(), CV_RGB(255, 255, 255));
            true_mat.copyTo(sub_square);

        } else {

            cv::Mat false_mat = cv::Mat::zeros(encode_bit_size_, encode_bit_size_, internal_frame_.type());
            false_mat.copyTo(sub_square);
        }

        sample_count >>= 1;
        column += encode_bit_size_;
    }
}

} /* namespace oat */
