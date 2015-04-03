/* 
 * File:   SMServer.h
 * Author: Jon Newman <jpnewman snail mit dot edu>
 *
 * Created on March 31, 2015, 6:37 PM
 */

#ifndef SMSERVER_H
#define	SMSERVER_H

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/named_condition.hpp>
#include <string>

template <class T>
class SMServer {
    
public:
    SMServer(std::string server_name);
    SMServer(const SMServer& orig);
    virtual ~SMServer();

    //const SMServer& operator=( const SMServer& other );

    void set_shared_object(T val);

protected:
    std::string name, shmem_name, shobj_name, mtx_name, cond_name;
    bool shared_write_object_created = false;
    boost::interprocess::named_condition serv_condition;
    boost::interprocess::managed_shared_memory shared_write_object;
    boost::interprocess::named_mutex serv_mutex;

    void createSharedObject(size_t bytes);

private:
    T* shared_object;

};

#endif	/* SMSERVER_H */

