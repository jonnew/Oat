//******************************************************************************
//* Copyright (c) Jon Newman (jpnewman at mit snail edu) 
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

#ifndef TESTPOSITION_H
#define	TESTPOSITION_H

#include <string>
#include <random>
#include <opencv2/core/mat.hpp>

#include "../../lib/datatypes/Position.h"
#include "../../lib/shmem/BufferedSMServer.h"

/**
 * Abstract base class by and Test Position class within the Simple Tracker project.
 * @param position_sink_name Name of the SINK to which test positions will be sent
 */
template <class T>
class TestPosition  {
    
public:
    
    TestPosition(std::string position_sink_name) : 
      position_sink(position_sink_name)
    , name(position_sink_name)
    , sample(0)
    , sample_period_in_seconds(0.02) { }

    // Test Positions can use a configuration file to specify parameters
    virtual void configure(const std::string& file_name, const std::string& key) = 0;
    
    // Test Positions simulate object position motion and publish to shared memory
    virtual void simulateAndServePosition(void) = 0;
    
    void stop(void) { position_sink.set_running(false); }
    
    // Accessors
    double get_sample_period(void) { return sample_period_in_seconds; }
    
protected:
    
    // Test position SINK name
    std::string name;

    // The test position SINK
    oat::BufferedSMServer<T> position_sink;
    
    // Test position sample number
    uint32_t sample;
    
    // Test positions update period
    double sample_period_in_seconds;
    
};

#endif	/* TESTPOSITION_H */

