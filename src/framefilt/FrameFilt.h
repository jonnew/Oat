//******************************************************************************
//* File:   FrameFilt.h
//* Author: Jon Newman <jpnewman snail mit dot edu>
//
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu) 
//* All right reserved.
//* This file is part of the Simple Tracker project.
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

#ifndef FRAMEFILT_H
#define	FRAMEFILT_H

#include <string>

#include "../../lib/shmem/MatClient.h"
#include "../../lib/shmem/MatServer.h"

class FrameFilt {
public:

    FrameFilt(const std::string source_name, const std::string sink_name) :
      frame_source(source_name)
    , frame_sink(sink_name) { }
    
    virtual ~FrameFilt() { }

    // Frame filters must be able to receive, filter, and serve frames
    virtual void filterAndServe(void) = 0;

    // Frame filters must be configurable
    virtual void configure(std::string config_file, std::string config_key) = 0;

    // Frame filters must be interruptable
    void stop(void) { frame_sink.set_running(false); }

private:

    // Frame filters have Mat client object for receiving frames
    shmem::MatClient frame_source;

    // Frame filters have Mat server for sending processed frames
    shmem::MatServer frame_sink;

};

#endif	/* FRAMEFILT_H */

