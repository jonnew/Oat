/* 
 * File:   SMServer.cpp
 * Author: Jon Newman <jpnewman snail mit dot edu>
 * 
 * Created on March 31, 2015, 6:37 PM
 */

#include "SMServer.h"

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <string>

using namespace boost::interprocess;

SMServer::SMServer(std::string block_name) {
    
    write_block_name = block_name;
    shared_memory_object::remove(write_block_name);
    
}

SMServer::SMServer(const SMServer& orig) {
}

SMServer::~SMServer() {
    
    shared_memory_object::remove("write_block_name");
}

SMServer::createSharedBlock(size_t bytes) {
    
    //Create a shared memory object.
      shared_write_object(create_only, write_block_name, read_write);

      //Set size
      shared_write_object.truncate(bytes);

      //Map the whole shared memory in this process
      write_region(shared_write_object, read_write);
      
      write_object_created = true;
}

