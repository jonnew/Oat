//******************************************************************************
//* File:   PositionSocket.h
//* Author: Jon Newman <jpnewman snail mit dot edu>
//*
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
#include <boost/asio.hpp>

#include "../../lib/shmem/SMClient.h"
#include "../../lib/datatypes/Position2D.h"

/**
 * Abstract position server.
 * All concrete position server types implement this ABC.
 */
class PositionSocket  {

public:
    
    PositionSocket(const std::string& position_source_name) : 
      name("posiserve[" + position_source_name + "->*]")
    , position_source(position_source_name) { }
      
    virtual ~PositionSocket() { }
    
    /**
     * Obtain position from SOURCE. Serve position to endpoint.
     * @return SOURCE end-of-stream signal. If true, this component should exit.
     */
    bool process(void) {

        // If source position valid, serve it
        if (position_source.getSharedObject(position))
            servePosition(position, position_source.get_current_time_stamp()); 
 
        // If server state is END, return true
        return (position_source.getSourceRunState() == oat::ServerRunState::END);  
    }
    
    // Accessors
    std::string get_name(void) const { return name; }
    
protected:    
    
    /**
     * Serve the position via specified IO protocol.
     * @param Position to serve.
     */
    virtual void servePosition(const oat::Position2D& current_position, const uint32_t sample) = 0;
    
    // IO service
    boost::asio::io_service io_service;

private:
    
    // Test position name
    const std::string name;
    
    // The test position SINK
    oat::SMClient<oat::Position2D> position_source;
    
    // The current position
    oat::Position2D position;
};

#endif	/* POSITIONSERVER_H */
