//******************************************************************************
//* File:   FrameFilter.h
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

#ifndef OAT_FRAMEFILT_H
#define	OAT_FRAMEFILT_H

#include <string>

#include "../../lib/base/Component.h"
#include "../../lib/base/Configurable.h"
#include "../../lib/datatypes/Frame.h"
#include "../../lib/shmemdf/Sink.h"
#include "../../lib/shmemdf/Source.h"

namespace oat {

class ColorConvert; // Forward decl.
namespace po = boost::program_options;

/**
 * @brief Abstract frame filter.
 */
class FrameFilter : public Component, public Configurable {

friend ColorConvert;

public:

    /**
     * @brief Abstract frame filter.
     * All concrete frame filter types implement this ABC.
     * @param frame_source_address Frame SOURCE node address
     * @param frame_sink_address Frame SINK node address
     */
    explicit FrameFilter(const std::string &frame_source_address,
                         const std::string &frame_sink_address);
    virtual ~FrameFilter() { };

    // Implement control interface
    virtual void connectToNode(void) override;
    int process(void) override;
    std::string name(void) const override { return name_; }
    oat::ComponentType type(void) const override { return oat::framefilter; };
    virtual int control(const char* msg) override;

protected:

    // Filter name
    const std::string name_;

    /**
     * Perform frame filtering. Override to implement filtering operation in
     * derived classes.
     * @param frame to be filtered
     */
    virtual void filter(cv::Mat &frame) = 0;

private:

    // Frame source
    const std::string frame_source_address_;
    oat::Source<oat::Frame> frame_source_;

    // Frame sink
    const std::string frame_sink_address_;
    oat::Sink<oat::Frame> frame_sink_;

    // Currently acquired, shared frame
    oat::Frame shared_frame_;
};

}      /* namespace oat */
#endif /* OAT_FRAMEFILT_H */
