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

#include <string>
#include <tuple>
#include <vector>
#include <boost/interprocess/sync/sharable_lock.hpp>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <cmath>

#include "../../lib/datatypes/Position2D.h"

#include "Decorator.h"

namespace oat {

Decorator::Decorator(const std::vector<std::string> &position_source_addresses,
                     const std::string &frame_source_address,
                     const std::string &frame_sink_address) :
  name_("decorator[" + frame_source_address+ "->" + frame_sink_address + "]")
, frame_source_address_(frame_source_address)
, frame_sink_address_(frame_sink_address)
{

    if (!position_source_addresses.empty()) {
        for (auto &addr : position_source_addresses) {

            oat::Position2D pos(addr);

            auto t = std::make_tuple(std::move(addr),
                                     std::move(pos),
                                     new oat::Source<oat::Position2D>());

            position_sources_.push_back(t);
        }
    } else {
        decorate_position = false;
    }
}

Decorator::~Decorator() {

    // Delete the memory pointed to by `new oat::Source<oat::Position2D>()`
    for (auto &pos : position_sources_)
        delete std::get<2>(pos);
}

void Decorator::connectToNodes() {

    // Connect to source node and retrieve cv::Mat parameters
    frame_source_.connect(frame_source_address_);
    oat::Source<oat::SharedCVMat>::MatParameters param =
            frame_source_.parameters();

    // Connect to position source nodes
    for (auto &pos : position_sources_)
        std::get<2>(pos)->connect(std::get<0>(pos));

    // Bind to sink sink node and create a shared cv::Mat
    frame_sink_.bind(frame_sink_address_, param.bytes);
    shared_frame_ = frame_sink_.retrieve(param.rows, param.cols, param.type);
}

bool Decorator::decorateFrame() {

    // 1. Get frame
    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sink to write to node
    if (frame_source_.wait() == oat::NodeState::END )
        return true;

    // Clone the shared frame
    internal_frame_ = frame_source_.clone();

    // Tell sink it can continue
    frame_source_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // 2. Get positions
    for (auto &pos : position_sources_) {

        // START CRITICAL SECTION //
        ////////////////////////////
        if (std::get<2>(pos)->wait() == oat::NodeState::END )
            return true;

        std::get<1>(pos) = std::get<2>(pos)->clone();

        std::get<2>(pos)->post();
        ////////////////////////////
        //  END CRITICAL SECTION  //
    }

    // Decorate frame
    drawSymbols();

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sources to read
    frame_sink_.wait();

    // Copy data to shared frame
    memcpy(shared_frame_.data, internal_frame_.data,
            internal_frame_.total() * internal_frame_.elemSize());

    // Tell sources there is new data
    frame_sink_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    // None of the sink's were at the END state
    return false;
}

void Decorator::drawSymbols() {

    if (decorate_position) {
        drawPosition();
        drawHeading();
        drawVelocity();

        if (print_region)
            printRegion();
    }

    if (print_timestamp)
        printTimeStamp();

    if (print_sample_number)
        printSampleNumber();

    if (encode_sample_number)
        encodeSampleNumber();
}

void Decorator::drawPosition() {

    for (auto pos : position_sources_) {
        if (std::get<1>(pos).position_valid) {
            cv::circle(internal_frame_, std::get<1>(pos).position, position_circle_radius, cv::Scalar(0, 0, 255), 2);
        }
    }
}

void Decorator::drawHeading() {
    for (auto pos : position_sources_) {
        if (std::get<1>(pos).position_valid && std::get<1>(pos).heading_valid) {
            cv::Point2d start = std::get<1>(pos).position - (heading_line_length * std::get<1>(pos).heading);
            cv::Point2d end = std::get<1>(pos).position + (heading_line_length * std::get<1>(pos).heading);

            // Draw arrow
            cv::line(internal_frame_, start, end, cv::Scalar(255, 0, 0), 2, 8);
            double angle = std::atan2((double) start.y - end.y, (double) start.x - end.x);
            start.x = end.x + heading_arrow_length * std::cos(angle + PI / 4);
            start.y = end.y + heading_arrow_length * std::sin(angle + PI / 4);
            cv::line(internal_frame_, start, end, cv::Scalar(255, 0, 0), 2, 8);
            start.x = end.x + heading_arrow_length * std::cos(angle - PI / 4);
            start.y = end.y + heading_arrow_length * std::sin(angle - PI / 4);
            cv::line(internal_frame_, start, end, cv::Scalar(255, 0, 0), 2, 8);
        }
    }
}

void Decorator::drawVelocity() {
    for (auto pos : position_sources_) {
        if (std::get<1>(pos).velocity_valid && std::get<1>(pos).position_valid) {
            cv::Point2d end = std::get<1>(pos).position + (velocity_scale_factor * std::get<1>(pos).velocity);
            cv::line(internal_frame_, std::get<1>(pos).position, end, cv::Scalar(0, 255, 0), 2, 8);
        }
    }
}

void Decorator::printRegion() {
    for (auto pos : position_sources_) {
        if (std::get<1>(pos).region_valid) {
            cv::Point text_origin(internal_frame_.cols - 100, 20);
            cv::putText(internal_frame_, "Region: " + std::string(std::get<1>(pos).region), text_origin, 1, font_scale, font_color);
        }
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
    cv::putText(internal_frame_, std::string(buffer), text_origin, 1, font_scale, font_color);
}

void Decorator::printSampleNumber() {

    cv::Point text_origin(10, internal_frame_.rows - 10);
    cv::putText(internal_frame_,
                std::to_string(frame_source_.write_number()),
                text_origin,
                1,
                font_scale,
                font_color);
}

/**
 * Encode the current sample number into the first row of the matrix
 */
void Decorator::encodeSampleNumber() {

    uint64_t sample_number = frame_source_.write_number();

    int column = 0;
    for (int shift = 0; shift < 32; shift++) {

        cv::Mat sub_square = internal_frame_.colRange(column, column + encode_bit_size).rowRange(0, encode_bit_size);

        if (sample_number & 0x1) {

            cv::Mat true_mat(encode_bit_size, encode_bit_size, internal_frame_.type(), CV_RGB(255, 255, 255));
            true_mat.copyTo(sub_square);

        } else {

            cv::Mat false_mat = cv::Mat::zeros(encode_bit_size, encode_bit_size, internal_frame_.type());
            false_mat.copyTo(sub_square);
        }

        sample_number >>= 1;
        column += encode_bit_size;
    }
}

} /* namespace oat */