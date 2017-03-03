//******************************************************************************
//* File:   PositionCombiner.h
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

#ifndef OAT_POSITIONCOMBINER_H
#define	OAT_POSITIONCOMBINER_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>

#include "../../lib/base/Component.h"
#include "../../lib/base/Configurable.h"
#include "../../lib/datatypes/Pose.h"
#include "../../lib/shmemdf/Helpers.h"
#include "../../lib/shmemdf/Sink.h"
#include "../../lib/shmemdf/Source.h"

namespace po = boost::program_options;

namespace oat {

/**
 * Abstract position combiner.
 * All concrete position combiner types implement this ABC.
 */
class PositionCombiner : public Component, public Configurable {

    using pvec_size_t = oat::NamedSourceList<oat::Position2D>::size_type;

public:
    // Component Interface
    oat::ComponentType type(void) const override { return oat::positioncombiner; };
    std::string name(void) const override { return name_; }

protected:
    // Combiner name
    std::string name_;

    /**
     * @brief Makes a list of position sources from a parsed program options
     * variable map.
     * @param vm Program options variable map containing the position source list.
     */
    void resolvePositionSources(const po::variables_map &vm);

    /**
     * Perform position combination.
     * @param sources SOURCE pose(s)
     * @return combined position
     */
    template <typename T>
    virtual oat::Pose combine(const T &source_poses) = 0;

    /**
     * Get the number of SOURCE positions.
     * @return number of SOURCE positions
     */
    int num_sources(void) const { return position_sources_.size(); };

private:
    // Component Interface
    virtual bool connectToNode(void) override;
    int process(void) override;


    // Pose SOURCES
    std::vector<oat::Position2D> positions_;
    oat::NamedSourceList<oat::Position2D> position_sources_;

    // Pose SINK
    std::string position_sink_address_;
    oat::Sink<oat::Position2D> position_sink_;

    // Currently acquired, shared pose
    oat::Pose *shared_pose_{nullptr};
};

}      /* namespace oat */
#endif /* OAT_POSITIONCOMBINER_H */
