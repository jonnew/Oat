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

SMServer::SMServer(std::string _name) {

	name = shared_object_name;
	shared_memory_object::remove(name);
}

SMServer::SMServer(const SMServer& orig) {
}

SMServer::~SMServer() {

	// This should be taken care of automatically by
	// remove_shared_memory_on_destroy
	shared_memory_object::remove(write_block_name);
}

SMServer::createSharedBlock(size_t bytes) {

	// Create a shared memory object
	shared_write_object(open_or_create, write_block_name, read_write);
	
	// Automatically cleanup shared memory when object 
	// is detroyed
	remove_on_destroy(shared_write_object);

	// Set size
	shared_write_object.truncate(bytes);

	// Map the whole shared memory in this processes'
	// address space
	write_region(shared_write_object, read_write);

	write_object_created = true;
}

