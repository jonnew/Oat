//******************************************************************************
//* File:   PositionGenerator.h
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

#ifndef OAT_POSITIONGENERATOR_H
#define	OAT_POSITIONGENERATOR_H

#include <chrono>
#include <limits>
#include <random>
#include <string>

#include <boost/program_options.hpp>
#include <opencv2/core/mat.hpp>

#include "../../lib/base/Component.h"
#include "../../lib/base/Configurable.h"
#include "../../lib/datatypes/Position2D.h"
#include "../../lib/shmemdf/Sink.h"

namespace po = boost::program_options;

namespace oat {

class PositionGenerator : public Component, public Configurable<false> {

public:
    /**
     * Abstract test position server.
     * All concrete test position server types implement this ABC and can be used
     * to server test positions with different motion characteristics to test
     * subsequent processing steps.
     * @param position_sink_name Position SINK to publish test positions
     */
    PositionGenerator(const std::string &position_sink_address);
    virtual ~PositionGenerator() { };

    // Component Interface
    oat::ComponentType type(void) const override { return oat::positiongenerator; };
    std::string name(void) const override { return name_; }

protected:
    /**
     * Generate test position.
     * @param position Generated position.
     * @return true if EOF has been genereated, false otherwise.
     */
    virtual bool generatePosition(oat::Position2D &position) = 0;

    // Test position sample clock
    bool enforce_sample_clock_ {false};
    std::chrono::high_resolution_clock clock_;
    std::chrono::duration<double> sample_period_in_sec_;
    std::chrono::high_resolution_clock::time_point start_, tick_;

    // Periodic boundaries in which simulated particle resides.
    cv::Rect_<double> room_ {0, 0, 100, 100};

    // Sample count specification
    uint64_t num_samples_ {std::numeric_limits<uint64_t>::max()};
    uint64_t it_ {0};

    /**
     * Configure the sample period
     * @param samples_per_second Sample period in seconds.
     */
    virtual void generateSamplePeriod(const double samples_per_second);

    /**
     * @brief Provide a copy of the base program options for derived
     * types that need it.
     * @return Base program options description.
     */
    po::options_description baseOptions(void) const;

private:
    // Component Interface
    virtual bool connectToNode(void) override;
    int process(void) override;

    // Test position name
    std::string name_;

    // Internally generated position
    oat::Position2D internal_position_ {"internal"};

    // Shared position
    oat::Position2D * shared_position_;

    // First position
    bool first_pos_ {true};

    // The test position SINK
    std::string position_sink_address_;
    oat::Sink<oat::Position2D> position_sink_;
};

}      /* namespace oat */
#endif /* OAT_POSITIONGENERATOR_H */
