/* 
 * File:   SMClient.cpp
 * Author: Jon Newman <jpnewman snail mit dot edu>
 * 
 * Created on March 31, 2015, 6:37 PM
 */

#include "SMClient.h"

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/sharable_lock.hpp>
#include <boost/interprocess/sync/interprocess_sharable_mutex.hpp>
#include <string>

using namespace boost::interprocess;

template<class SyncType>
SMClient<SyncType>::SMClient(std::string server_name) :
  name(server_name)
, shmem_name(server_name.append("_sh_mem"))
, shobj_name(server_name.append("_sh_obj"))
{ }

template<class SyncType>
SMClient<SyncType>::SMClient(const SMClient& orig) {
}

template<class SyncType>
SMClient<SyncType>::~SMClient() {

    // Clean up sync objects
    shared_object->cond_var.notify_all();
}

template<class SyncType>
void SMClient<SyncType>::findSharedObject() {

    try {

        // Allocate shared memory
        shared_memory = managed_shared_memory(open_only, shmem_name.c_str());

        // Make the shared object
        shared_object = shared_memory.find<SyncType>(shobj_name.c_str()).first;

    } catch (bad_alloc &ex) {
        std::cerr << ex.what() << '\n';
    }

    shared_read_object_created = true;
}

