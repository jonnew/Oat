//******************************************************************************
//* File:   PositionUDPServer.h
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
#ifndef POSITIONUDPSERVER_H
#define	POSITIONUDPSERVER_H

#include <boost/asio.hpp>

#include "PositionServer.h"

using baiu = boost::asio::ip::udp;

class PositionUDPServer : PositionServer{
    
public:
    PositionUDPServer(const std::string& position_source_name, unsigned short port=9000);

private:

    //
    unsigned short port;
    
    boost::asio::io_service io_service;
    baiu::socket socket;
    baiu::endpoint remote_endpoint;
};

#endif	/* POSITIONUDPSERVER_H */

