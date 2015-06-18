//******************************************************************************
//* File:   SyncSharedMemoryManager.h
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

#ifndef SHAREDMEMORYMANAGER_H
#define	SHAREDMEMORYMANAGER_H

#include <atomic>

namespace oat {

    enum class ServerRunState {
        END = -1,
        UNDEFINED = 0,
        RUNNING = 1,
        ERROR = 2
    };


    class SharedMemoryManager {
    public:

        SharedMemoryManager() :
          server_state(ServerRunState::UNDEFINED)
        , client_reference_count(0) { }

        // These operations are atomic
        void set_server_state(ServerRunState value) { server_state = value; }
        ServerRunState get_server_state(void) const { return server_state; }
        size_t decrementClientRefCount() { return --client_reference_count; }
        size_t incrementClientRefCount() { return ++client_reference_count; }
        size_t get_client_ref_count(void) const { return client_reference_count; }
        
    private:

        std::atomic<ServerRunState> server_state;

        // Number of clients sharing this shared memory
        std::atomic<size_t> client_reference_count;

    };

} // namespace oat


#endif	/* SHAREDMEMORYMANAGER_H */

