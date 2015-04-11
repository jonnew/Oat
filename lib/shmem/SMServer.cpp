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

#include "SMServer.h"

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <string>

using namespace boost::interprocess;

template<class SyncType>
SMServer<SyncType>::SMServer(std::string sink_name) :
  name(sink_name)
, shmem_name(sink_name + "_sh_mem")
, shobj_name(sink_name + "_sh_obj")
{ }

template<class SyncType>
SMServer<SyncType>::SMServer(const SMServer& orig) {
}

template<class SyncType>
SMServer<SyncType>::~SMServer() {

    // Remove_shared_memory on object destruction
    shared_object->cond_var.notify_all();
    shared_memory_object::remove(shmem_name.c_str());
}

template<class SyncType>
void SMServer<SyncType>::createSharedObject( ) {

    try {

        // Clean up any potential leftovers
        shared_memory_object::remove(shmem_name.c_str());

        // Allocate shared memory
        shared_memory = managed_shared_memory(open_or_create, shmem_name.c_str(), sizeof(SyncType) + 1024);

        // Make the shared object
        shared_object = shared_memory.find_or_construct<SyncType>(shobj_name.c_str())();
        
        // Set the ready flag
        shared_object->ready = true;

    } catch (bad_alloc &ex) {
        std::cerr << ex.what() << '\n';
    }
}

template<class SyncType>
void SMServer<SyncType>::set_value(SyncType value) {
    
    if (!value->ready) {
       createSharedObject( ); 
       value->ready = true; 
    }
    
    // Exclusive scoped_lock on the shared_mat_header->mutex
    scoped_lock<interprocess_sharable_mutex> lock(value->mutex);
    
    // Perform write in shared memory
    *shared_object = value;
    
    // Notify all client processes they can now access the data
    value->cond_var.notify_all();
    
} // Scoped lock is released
