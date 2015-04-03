/* 
 * File:   SMClient.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on March 31, 2015, 6:37 PM
 */

#ifndef SMCLIENT_H
#define	SMCLIENT_H

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/mapped_region.hpp>
//#include <boost/interprocess/sync/named_sharable_mutex.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/named_condition.hpp>
#include <string>

template <class T>
class SMClient {
    
public:
    SMClient(std::string server_name);
    SMClient(const SMClient& orig);
    virtual ~SMClient();

    //const SMServer& operator=( const SMServer& other );

    //void get_shared_object(T* val);

protected:

    T* shared_object;
    
    std::string name, shmem_name, shobj_name, mtx_name, cond_name;
    bool shared_read_object_created = false;
    boost::interprocess::named_condition client_condition;
    boost::interprocess::managed_shared_memory shared_memory;
    boost::interprocess::named_mutex client_mutex;
    
    void findSharedObject(void);
    

};

#endif	/* SMCLIENT_H */

