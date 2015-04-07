/* 
 * File:   SMServer.cpp
 * Author: Jon Newman <jpnewman snail mit dot edu>
 * 
 * Created on March 31, 2015, 6:37 PM
 */

#include "SMServer.h"

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <string>

using namespace boost::interprocess;

template<class SyncType>
SMServer<SyncType>::SMServer(std::string server_name) :
name(server_name)
, shmem_name(name.append("_sh_mem"))
, shobj_name(name.append("_sh_obj"))
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
void SMServer<SyncType>::createSharedObject(size_t bytes) {

    try {

        // Clean up any potential leftovers
        shared_memory_object::remove(shmem_name.c_str());
        //        named_sharable_mutex::remove(mtx_name.c_str());
        //        named_condition::remove(cond_name.c_str());

        // Allocate shared memory
        shared_memory = managed_shared_memory(open_or_create, shmem_name.c_str(), bytes);

        // Make the shared object
        shared_object = shared_memory.find_or_construct<SyncType>(shobj_name.c_str())();

    } catch (bad_alloc &ex) {
        std::cerr << ex.what() << '\n';
    }

    shared_write_object_created = true;
}

