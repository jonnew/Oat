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

#ifndef SYNCSHAREDMEMORYOBJECT_H
#define	SYNCSHAREDMEMORYOBJECT_H

#include <utility>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>

#include "../datatypes/Position.h"
#include "../datatypes/Position2D.h"

namespace oat {

    template <class T>
    class SyncSharedMemoryObject {
    public:

        SyncSharedMemoryObject(void) :
          mutex(1)
        , write_barrier(0)
        , read_barrier(0)
        , new_data_barrier(0)
        , number_of_clients(0)
        , client_read_count(0)
        , sample_number(0) { }

        // Semaphores used to synchronize access to the shared object
        boost::interprocess::interprocess_semaphore mutex;
        boost::interprocess::interprocess_semaphore write_barrier;
        boost::interprocess::interprocess_semaphore read_barrier;
        boost::interprocess::interprocess_semaphore new_data_barrier;
        
        size_t number_of_clients;
        size_t client_read_count;
        
        // Time keeping TODO: use accessors!
        uint32_t sample_number; // Sample number of this position, respecting buffer overruns

        // Write/read access to shared object
        
        /**
         * Move object into shared memory slot. 
         * @param value Value to be moved to shared memory. Value
         * is left in a valid but unspecified state after this operation.
         */
        void writeSample(uint32_t sample, T value) { sample_number = sample; object = std::move(value); }
        T get_value(void) const { return object; } // Read-only (for clients, forces copy if they want to mess with object)

    private:
        
        // Shared object
        T object;

    };
}

// Explicit instantiations
template class oat::SyncSharedMemoryObject<oat::Position2D>;

#endif	/* SYNCSHAREDMEMORYOBJECT_H */

