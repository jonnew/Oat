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
#include <ctime>
#include <exception>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "../../lib/datatypes/Pose.h"
#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/Pallet.h"
#include "../../lib/utility/TOMLSanitize.h"
#include "../../lib/utility/make_unique.h"

#include "Decorator.h"

namespace oat {

Decorator::Decorator(const std::string &frame_source_address,
                     const std::string &frame_sink_address)
: name_("decorator[" + frame_source_address+ "->" + frame_sink_address + "]")
, frame_source_address_(frame_source_address)
, frame_sink_address_(frame_sink_address)
{
    // Nothing
}

po::options_description Decorator::options() const
{
    po::options_description local_opts;
    local_opts.add_options()
        ("pose-sources,p", po::value< std::vector<std::string> >()->multitoken(),
        "The name of pose-sources SOURCE(s) used to draw object pose-sources markers.\n")
        ("timestamp,t", "Write the current date and time on each frame.\n")
        ("sample,s", "Write the frame sample number on each frame.\n")
        ("sample-code,S", "Write the binary encoded sample on the corner of each frame.\n")
        ("region,R", "Write region information on each frame "
        "if there is a pose-sources stream that contains it.\n")
        ("history,h", "Display pose-sources history.\n")
        ("font-scale,f", po::value<double>(),
         "Scale all font sizes by this value. Defaults to 1.0.\n")
        ("invert-font,i", "Invert font color.\n")
        ("marker-size,l", po::value<double>(),
         "Size of pose marker in whatever distance units the pose's position is "
         "expressed in.\n")
        ;

    return local_opts;
}

void Decorator::applyConfiguration(const po::variables_map &vm,
                                   const config::OptionTable &config_table)
{
    // Position sources
    // NOTE: not settable via configuration file
    if (vm.count("pose-sources"))  {

        auto p_source_addrs = vm["pose-sources"].as<std::vector<std::string>>();

        // Setup pose-sources sources
        if (!p_source_addrs.empty()) {
            for (auto &addr : p_source_addrs) {

                pose_sources_.push_back(oat::NamedSource<oat::Pose>(
                    addr, oat::make_unique<oat::Source<oat::Pose>>()));
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

    // Font scale
    oat::config::getNumericValue<double>(
        vm, config_table, "font-scale", font_scale_, 0);

    // Invert font
    bool invert_font;
    if (oat::config::getValue<bool>(vm, config_table, "invert-font", invert_font))
        cv::subtract(cv::Scalar(255, 255, 255), font_color_, font_color_);

    // Marker size
    oat::config::getNumericValue<double>(
        vm, config_table, "marker-size", marker_size_, 0);

}

bool Decorator::connectToNode()
{
    // Examine sample period of sources to make sure they are the same
    double sample_rate_hz;
    std::vector<double> all_ts;

    // Establish our a slots in the frame and positions sources
    frame_source_.touch(frame_source_address_);

    for (auto &ps : pose_sources_)
        ps.source->touch(ps.name);

    // Wait for synchronous start with sink when it binds the node
    if (frame_source_.connect() != SourceState::CONNECTED)
        return false;

    for (auto &ps : pose_sources_) {
        ps.source->connect();
        all_ts.push_back(ps.source->retrieve()->sample_period_sec());
    }

    // Get frame meta data to format sink
    oat::Source<oat::Frame>::FrameParams param = frame_source_.parameters();

    // Bind to sink sink node and create a shared frame
    frame_sink_.bind(frame_sink_address_, param.bytes);
    shared_frame_
        = frame_sink_.retrieve(param.rows, param.cols, param.type, param.color);
    all_ts.push_back(shared_frame_.sample_period_sec());

    if (!oat::checkSamplePeriods(all_ts, sample_rate_hz))
        std::cerr << oat::Warn(oat::inconsistentSampleRateWarning(sample_rate_hz));

    // Set drawing parameters based on frame dimensions
    encode_bit_size_
        = std::ceil(param.cols / 3 / sizeof(uint64_t) / 8); //sizeof(frame.sample_count())

    // If we are drawing positions, get ready for that
    if (decorate_position_) {
        poses_found_.push_back(false);
        history_frame_
            = cv::Mat::zeros(shared_frame_.size(), shared_frame_.type());
    }

    return true;
}

int Decorator::process()
{
    // 1. Get frame
    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sink to write to node
    if (frame_source_.wait() == oat::NodeState::END)
        return 1;

    // Clone the shared frame
    oat::Frame frame;
    frame_source_.copyTo(frame);

    // Tell sink it can continue
    frame_source_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // 2. Get poses
    std::vector<oat::Pose> poses;
    poses.reserve(pose_sources_.size());
    for (pvec_size_t i = 0; i !=  pose_sources_.size(); i++) {

        // START CRITICAL SECTION //
        ////////////////////////////
        if (pose_sources_[i].source->wait() == oat::NodeState::END)
            return 1;

        poses.push_back(pose_sources_[i].source->clone());

        pose_sources_[i].source->post();
        ////////////////////////////
        //  END CRITICAL SECTION  //
    }

    // Decorate frame
    decorate(frame, poses);

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sources to read
    frame_sink_.wait();

    frame.copyTo(shared_frame_);

    // Tell sources there is new data
    frame_sink_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // None of the sink's were at the END state
    return 0;
}

oat::CommandDescription Decorator::commands()
{
    const oat::CommandDescription commands{
        {"clear", "Clear path history."}
    };

    return commands;
}

void Decorator::applyCommand(const std::string &command)
{
    if (command == "clear")
        history_frame_ = cv::Scalar::all(0);
}

void Decorator::decorate(oat::Frame &frame, const std::vector<oat::Pose> &poses)
{
    if (decorate_position_) {

        drawPose(frame, poses);
        if (print_region_)
            printRegion(frame, poses);
    }

    if (print_timestamp_)
        printTimeStamp(frame);

    if (print_sample_number_)
        printSampleNumber(frame);

    if (encode_sample_number_)
        encodeSampleNumber(frame);
}

void Decorator::drawPose(oat::Frame &frame, const std::vector<oat::Pose> &poses)
{
    cv::Mat symbol_frame =
        cv::Mat::zeros(frame.size(), frame.type());

    oat::RGB<MixedPallet> pallet;

    for (size_t i = 0; i < poses.size(); i++) {

        if (poses[i].found) {

            std::vector<cv::Point3f> axis_3d;
            axis_3d.push_back(cv::Point3f(0, 0, 0));
            axis_3d.push_back(cv::Point3f(marker_size_, 0, 0));
            axis_3d.push_back(cv::Point3f(0, marker_size_, 0));
            axis_3d.push_back(cv::Point3f(0, 0, marker_size_));
            std::vector<cv::Point2f> axis_2d;
            cv::projectPoints(axis_3d,
                              poses[i].orientation<cv::Vec3d>(),
                              poses[i].position<cv::Vec3d>(),
                              camera_matrix_,
                              dist_coeff_,
                              axis_2d);

            auto center = axis_2d[0];

            // This color
            cv::Scalar col = pallet.next();

            // Draw axis or point
            if (poses[i].orientation_dof >= Pose::DOF::Two) {
                cv::line(frame, center, axis_2d[1], cv::Scalar(0, 0, 255), line_thickness_);
                cv::line(frame, center, axis_2d[2], cv::Scalar(0, 255, 0), line_thickness_);
                if (poses[i].orientation_dof == Pose::DOF::Three)
                    cv::line(frame, center, axis_2d[3], cv::Scalar(255, 0, 0), line_thickness_);
            } else {
                cv::circle(frame, center, marker_size_ / 2, col, line_thickness_);
            }

            if (show_position_history_ && poses_found_[i])
                cv::circle(history_frame_, center, 1, col, 1);

            poses_found_[i] = true;
        } else {
            poses_found_[i] = false;
        }
    }

    // TODO: Following is inefficient and it shows in the performance testing
    if (show_position_history_)
        symbol_frame += history_frame_;

    cv::Mat result_frame =
        cv::Mat::zeros(frame.size(), frame.type());
    cv::addWeighted(frame,
                    1 - symbol_alpha_,
                    symbol_frame,
                    symbol_alpha_,
                    0.0,
                    result_frame);

    cv::Mat mask;
    const cv::Scalar zero(0);
    cv::inRange(symbol_frame, zero, zero, mask);
    frame.setTo(zero, mask == 0);
    result_frame.setTo(zero, mask);
    frame += result_frame;
}

void Decorator::printRegion(oat::Frame &frame,
                            const std::vector<oat::Pose> &poses)
{
    // Create display string
    std::string reg_text;

    if (pose_sources_.size() == 1)
        reg_text = "Region:";
    else
        reg_text = "Regions:";

    // Calculate text origin based upon message size
    int baseline = 0;
    cv::Size ts = cv::getTextSize(
        reg_text, font_type_, font_scale_, font_thickness_, &baseline);

    cv::Point text_origin(10, ts.height);
    cv::putText(frame, reg_text, text_origin, font_thickness_, font_scale_, font_color_);

    // Add "ID: region" information
    oat::RGB<MixedPallet> pallet;
    for (const auto &p : poses) {
        if (p.in_region)
            reg_text = " " + std::string(p.region);
        else
            reg_text = " ?";

        text_origin.y += ts.height + 2;
        cv::putText(frame,
                    reg_text, text_origin,
                    font_thickness_,
                    font_scale_,
                    pallet.next());
    }
}

void Decorator::printTimeStamp(oat::Frame &frame)
{
    std::time_t raw_time;
    struct tm * time_info;
    char buffer[80];

    std::time(&raw_time);
    time_info = std::localtime(&raw_time);
    size_t n = std::strftime(buffer, 80, "%c", time_info);
    const auto text = std::string(buffer).substr(0, n);

    int baseline = 0;
    cv::Size ts = cv::getTextSize(
        text, font_type_, font_scale_, font_thickness_, &baseline);

    // NB: I have no idea why I have to divide the width by 2 here
    cv::Point text_origin(frame.cols - ts.width / 2 - 10, frame.rows - 10);
    cv::putText(
        frame, std::string(buffer), text_origin, 1, font_scale_, font_color_);
}

void Decorator::printSampleNumber(oat::Frame &frame)
{
    cv::Point text_origin(10, frame.rows - 10);
    cv::putText(frame,
                std::to_string(frame.sample_count()),
                text_origin,
                1,
                font_scale_,
                font_color_);
}

void Decorator::encodeSampleNumber(oat::Frame &frame)
{
    uint64_t sample_count = frame.sample_count();
    int column = frame.cols - 64 * encode_bit_size_;

    if (column < 0)
        throw std::runtime_error(
            "Binary counter is too large for frame. "
            "Use more x-dim pixels or turn binary counter off.");

    for (int shift = 0; shift < 64; shift++) {

        cv::Mat sub_square = frame.colRange(column, column + encode_bit_size_)
                                 .rowRange(0, encode_bit_size_);

        if (sample_count & 0x1) {

            cv::Mat true_mat(encode_bit_size_,
                             encode_bit_size_,
                             frame.type(),
                             oat::RGB<oat::Roygbiv>::white());
            true_mat.copyTo(sub_square);

        } else {

            cv::Mat false_mat = cv::Mat::zeros(
                encode_bit_size_, encode_bit_size_, frame.type());
            false_mat.copyTo(sub_square);
        }

        sample_count >>= 1;
        column += encode_bit_size_;
    }
}

} /* namespace oat */
