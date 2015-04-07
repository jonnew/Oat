/* 
 * File:   SMServer.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on March 31, 2015, 6:37 PM
 */

#ifndef SMSERVER_H
#define	SMSERVER_H

#include <boost/interprocess/managed_shared_memory.hpp>
#include <string>

template <class SyncType>
class SMServer {
    
public:
    SMServer(std::string server_name);
    SMServer(const SMServer& orig);
    virtual ~SMServer();

protected:
    
    SyncType* shared_object;
    
    std::string name, shmem_name, shobj_name, mtx_name, cond_name;
    bool shared_write_object_created = false;
    boost::interprocess::managed_shared_memory shared_memory;

    void createSharedObject(size_t bytes);
    void set_shared_object(SyncType val);
   
};

#endif	/* SMSERVER_H */

