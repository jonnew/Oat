/* 
 * File:   SMClient.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on March 31, 2015, 6:37 PM
 */

#ifndef SMCLIENT_H
#define	SMCLIENT_H

#include <boost/interprocess/managed_shared_memory.hpp>
#include <string>

template <class SyncType>
class SMClient {
public:
    SMClient(std::string server_name);
    SMClient(const SMClient& orig);
    virtual ~SMClient();

protected:
    
    SyncType* shared_object;

    std::string name, shmem_name, shobj_name, mtx_name, cond_name;
    bool shared_read_object_created = false;
    boost::interprocess::managed_shared_memory shared_memory;

    void findSharedObject(void);

};

#endif	/* SMCLIENT_H */

