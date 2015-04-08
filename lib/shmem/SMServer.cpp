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
  srv_name(sink_name)
, srv_shmem_name(srv_name + "_sh_mem")
, srv_shobj_name(srv_name + "_sh_obj")
{ }

template<class SyncType>
SMServer<SyncType>::SMServer(const SMServer& orig) {
}

template<class SyncType>
SMServer<SyncType>::~SMServer() {

    // Remove_shared_memory on object destruction
    srv_shared_object->cond_var.notify_all();
    shared_memory_object::remove(srv_shmem_name.c_str());
}

template<class SyncType>
void SMServer<SyncType>::createSharedObject(size_t bytes) {

    try {

        // Clean up any potential leftovers
        shared_memory_object::remove(srv_shmem_name.c_str());
        //        named_sharable_mutex::remove(mtx_name.c_str());
        //        named_condition::remove(cond_name.c_str());

        // Allocate shared memory
        srv_shared_memory = managed_shared_memory(open_or_create, srv_shmem_name.c_str(), bytes);

        // Make the shared object
        srv_shared_object = srv_shared_memory.find_or_construct<SyncType>(srv_shobj_name.c_str())();

    } catch (bad_alloc &ex) {
        std::cerr << ex.what() << '\n';
    }

    srv_shared_write_object_created = true;
}

