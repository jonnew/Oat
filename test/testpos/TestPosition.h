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
#include "../../lib/shmem/SMServer.h"

#define DT 0.02 // TODO: Config

/**
 * Abstract base class by and Test Position class within the Simple Tracker project.
 * @param position_sink_name Name of the SINK to which test positions will be sent
 */
class TestPosition  {
    
public:
    
    TestPosition(std::string position_sink_name) : 
      position_sink(position_sink_name)
    , name(position_sink_name) { }

    // Test Positions can use a configuration file to specify parameters
    //virtual void configure(std::string file_name, std::string key) = 0;
    
    // Test Positions simulate object position motion and publish to shared memory
    virtual void simulateAndServePosition(void) = 0;
    
    void stop(void) { position_sink.set_running(false); }
    
protected:
    
    // Test position SINK name
    std::string name;

    // The test position SINK
    shmem::SMServer<datatypes::Position> position_sink;
    
};

#endif	/* TESTPOSITION_H */

