//******************************************************************************
//* File:   PositionServer.h
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

#ifndef POSITIONSERVER_H
#define	POSITIONSERVER_H

#include <string>

#include "../../lib/shmem/SMClient.h"

/**
 * Abstract position server.
 * All concrete position server types implement this ABC.
 */
template <class T>
class PositionServer  {

public:
    
    PositionServer(const std::string& position_source_name) : 
      name("posiserve[" + position_source_name + "->*]")
    , position_source(position_source_name) { }
      
    virtual ~PositionServer() { }
    
protected:    
    
    /**
     * Serve the position via specified IO protocol.
     * @param Position to serve.
     */
    virtual void servePosition(T) = 0;
    
private:
    
    // Test position name
    std::string name;

    // The test position SINK
    oat::SMClient<T> position_source;
}


#endif	/* POSITIONSERVER_H */

