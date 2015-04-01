/* 
 * File:   SMServer.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on March 31, 2015, 6:37 PM
 */

#ifndef SMSERVER_H
#define	SMSERVER_H

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <string>

class SMServer {
public:
    SMServer(std::string block_name);
    SMServer(const SMServer& orig);
    virtual ~SMServer();

    void createSharedBlock(size_t bytes);
    void openSharedBlock(std::string block_to_open);

protected:

    std::string name;
    bool shared_object_created = false;
    boost::interprocess::managed_shared_memory shared_write_object;
	boost::interprocess::remove_shared_memory_on_destroy remove_on_destroy;	
    boost::interprocess::mapped_region write_region;
    std::vector<boost::interprocess::shared_memory_object> read_blocks;

};

#endif	/* SMSERVER_H */

