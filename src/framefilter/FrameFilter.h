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
#include "../../lib/datatypes/Frame2.h"
#include "../../lib/shmemdf/Sink2.h"
#include "../../lib/shmemdf/Source.h"

namespace oat {

namespace po = boost::program_options;

class FrameFilter : public Component {

public:
    /**
     * @brief Abstract frame filter.
     * All concrete frame filter types implement this ABC.
     * @param frame_source_address Frame SOURCE node address
     * @param frame_sink_address Frame SINK node address
     */
    FrameFilter(const std::string &frame_source_address,
                const std::string &frame_sink_address);
    virtual ~FrameFilter() { };

    // Component Interface
    oat::ComponentType type(void) const override { return oat::framefilter; };

protected:
    /**
     * Perform frame filtering. Override to implement filtering operation in
     * derived classes.
     * @param frame to be filtered
     */
    virtual void filter(oat::Frame &frame) = 0;

    // Check frame pixel color type
    // Default to not caring about color
    virtual bool checkPixelColor(oat::Pixel::Color c)
    {
        (void)c; // Override unused variable warning
        return true;
    }

private:
    // Component Interface
    virtual bool connectToNode(void) override;
    int process(void) override;

    // Frame source
    std::unique_ptr<oat::SharedFrame> sh_frame_;
    oat::FrameSource frame_source_;

    // Frame sink
    oat::FrameSink frame_sink_;
};

}      /* namespace oat */
#endif /* OAT_FRAMEFILT_H */
