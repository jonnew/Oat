/* 
 * File:   SMClient.cpp
 * Author: Jon Newman <jpnewman snail mit dot edu>
 * 
 * Created on March 31, 2015, 6:37 PM
 */

#include "SMClient.h"

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <string>

using namespace boost::interprocess;

template<class T>
SMClient<T>::SMClient(std::string server_name) : 
      name(server_name) 
    , shmem_name(server_name.append("_sh_mem"))
    , shobj_name(server_name.append("_sh_obj"))
    , mtx_name(server_name.append("_mtx"))
    , cond_name(server_name.append("_cond"))
    , client_mutex(open_or_create, mtx_name.c_str())
    , client_condition(open_or_create, cond_name.c_str()) {
}

template<class T>
SMClient<T>::SMClient(const SMClient& orig) {
}

template<class T>
SMClient<T>::~SMClient() {

    // Clean up sync objects
    client_condition.notify_all();
    named_mutex::remove(mtx_name.c_str());
    named_condition::remove(cond_name.c_str());
}

template<class T>
void SMClient<T>::findSharedObject() {

    try {

        // Allocate shared memory
        shared_read_object = managed_shared_memory(open_only, shmem_name.c_str());

        // Make the shared object
        shared_object = shared_read_object.construct<T>(shobj_name.c_str())();

    } catch (boost::interprocess::bad_alloc &ex) {
        std::cerr << ex.what() << '\n';
    }

    shared_read_object_created = true;
}

template<class T>
void SMClient<T>::get_shared_object(T* val) {
    
    {
        // Lock access to the shared object
        scoped_lock<named_mutex> lock{client_mutex};
        
        *shared_object = *val; 
        
        client_condition.notify_all();
        client_condition.wait(lock);
    } 
}

