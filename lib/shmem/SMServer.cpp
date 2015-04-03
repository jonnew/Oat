/* 
 * File:   SMServer.cpp
 * Author: Jon Newman <jpnewman snail mit dot edu>
 * 
 * Created on March 31, 2015, 6:37 PM
 */

#include "SMServer.h"

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <string>

using namespace boost::interprocess;

template<class T>
SMServer<T>::SMServer(std::string server_name) : 
      name(server_name) 
    , shmem_name(name.append("_sh_mem"))
    , shobj_name(name.append("_sh_obj"))
    , mtx_name(name.append("_mtx"))
    , cond_name(name.append("_cond"))
    , serv_mutex(open_or_create, mtx_name.c_str())
    , serv_condition(open_or_create, cond_name.c_str())
{

    
}

template<class T>
SMServer<T>::SMServer(const SMServer& orig) { }

template<class T>
SMServer<T>::~SMServer() {

    // Remove_shared_memory on object destruction
    serv_condition.notify_all();
    shared_memory_object::remove(shmem_name.c_str());
    named_mutex::remove(mtx_name.c_str());
    named_condition::remove(cond_name.c_str());
}

template<class T>
void SMServer<T>::createSharedObject(size_t bytes) {

    try {
        
        // Clean up any potential leftovers
        shared_memory_object::remove(shmem_name.c_str());
        named_mutex::remove(mtx_name.c_str());
        named_condition::remove(cond_name.c_str());

        // Allocate shared memory
        shared_write_object = managed_shared_memory(open_or_create, shmem_name.c_str(), bytes);

        // Make the shared object
        shared_object = shared_write_object.construct<T>(shobj_name.c_str())();

    } catch (boost::interprocess::bad_alloc &ex) {
        std::cerr << ex.what() << '\n';
    }

    shared_write_object_created = true;
}

template<class T>
void SMServer<T>::set_shared_object(T val) {
    
    // Lock access to the shared object
    scoped_lock<named_mutex> lock{serv_mutex};
    {

        *shared_object = val; 
        
        serv_condition.notify_all();
        serv_condition.wait(lock);
    } 
}

