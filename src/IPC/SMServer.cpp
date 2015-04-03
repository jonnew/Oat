/* 
 * File:   SMServer.cpp
 * Author: Jon Newman <jpnewman snail mit dot edu>
 * 
 * Created on March 31, 2015, 6:37 PM
 */

#include "SMServer.h"

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <string>

using namespace boost::interprocess;

SMServer::SMServer(std::string shared_object_name) {

    name = shared_object_name;
    
}

SMServer::SMServer(const SMServer& orig) {
}

SMServer::~SMServer() {

    // Remove_shared_memory on object destruction
    shared_memory_object::remove(name);
}

SMServer::createSharedObject(size_t bytes) {

    try {
        shared_memory_object::remove(name);
        shared_write_object{open_or_create, name, bytes};
        T *shared_object = shared_write_object.construct<T>("w_" + name)();
    } catch (boost::interprocess::bad_alloc &ex) {
        std::cerr << ex.what() << '\n';
    }

    shared_write_object_created = true;
}

