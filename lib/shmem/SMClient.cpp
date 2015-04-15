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

#include "SMClient.h"

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/sharable_lock.hpp>
#include <boost/interprocess/sync/interprocess_sharable_mutex.hpp>
#include <string>

using namespace boost::interprocess;

template<class SyncType, class IOType>
SMClient<SyncType, IOType>::SMClient(std::string source_name) :
  name(source_name)
, shmem_name(source_name + "_sh_mem")
, shobj_name(source_name + "_sh_obj") {
}

template<class SyncType, class IOType>
SMClient<SyncType, IOType>::SMClient(const SMClient& orig) {
}

template<class SyncType, class IOType>
SMClient<SyncType, IOType>::~SMClient() {

    // Clean up sync objects
    shared_object->new_data_condition.notify_all();
}

template<class SyncType, class IOType>
void SMClient<SyncType, IOType>::findSharedObject() {

    try {

        // Allocate shared memory
        cli_shared_memory = managed_shared_memory(open_only, shmem_name.c_str());

        // Find the object in shared memory
        shared_object = cli_shared_memory.find<SyncType>(shobj_name.c_str()).first;
        shared_object_found = true;

    } catch (interprocess_exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << "  This is likely due to the SOURCE, \"" << name << "\", not being started.\n";
        std::cerr << "  Did you start the SOURCE, \"" << name << "\", before staring this client?" << std::endl;
        exit(EXIT_FAILURE); // TODO: exit does not unwind the stack to take care of destructing shared memory objects
    }

    lock = makeLock();
}

template<class SyncType, class IOType>
sharable_lock<interprocess_sharable_mutex> SMClient<SyncType, IOType>::makeLock(void) {
    sharable_lock<interprocess_sharable_mutex> sl(shared_object->mutex); // defer_lock
    return sl;
}