//******************************************************************************
//* File:   PositionWriter.h
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
//*****************************************************************************

#ifndef OAT_POSITIONWRITER_H
#define OAT_POSITIONWRITER_H

#include "Writer.h"

#include <rapidjson/filewritestream.h>
#include <rapidjson/prettywriter.h>

#include "../../lib/datatypes/Position2D.h"

namespace oat {
namespace blf = boost::lockfree;

// Constants
static constexpr int POSITION_WRITE_BUFFER_SIZE {65536};

/**
 * Position stream file writer.
 */
class PositionWriter : public Writer<oat::Position2D> {

    // Inherit constructor
    using Writer<oat::Position2D>::Writer;

public:

    ~PositionWriter();

    void write(void) override;

    void initialize(const std::string &source_name,
                    const oat::Position2D &p) override;

    // Accessors
    void set_verbose_file(const bool value) { verbose_file_ = value; }

private:

    /**
     * @brief Determines if indeterminate position data fields should be
     * written in spite of being indeterminate for sample parsing ease? e.g.
     * Should we write pos_xy when pos_ok = false?
     */
    bool verbose_file_ {true};

    // Timestamp clock
    std::chrono::system_clock clock_;
    std::chrono::system_clock::time_point start_;

    // Position file
    // TODO: Position specialization
    FILE * fd_ {nullptr};
    char position_write_buffer[POSITION_WRITE_BUFFER_SIZE];
    std::unique_ptr<rapidjson::FileWriteStream> file_stream_;
    rapidjson::PrettyWriter<rapidjson::FileWriteStream> json_writer_ {*file_stream_};
};

}      /* namespace oat */
#endif /* OAT_POSITIONWRITER_H */
