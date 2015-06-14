//******************************************************************************
//* File:   Signals.h
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

#ifndef SIGNALS_H
#define	SIGNALS_H

#include <atomic>

namespace oat {

    enum class ServerRunState {
        
        END = -1, 
        UNDEFINED = 0, 
        RUNNING = 1, 
        ERROR = 2
    };
    
    class ServerState {
        
    public:
        ServerRunState get_state(void) const { return server_state; }  
        void set_state(ServerRunState value) { server_state = value; }
        
    private:
        std::atomic<ServerRunState> server_state {ServerRunState::UNDEFINED};
    };
    
}

#endif	/* SIGNALS_H */

