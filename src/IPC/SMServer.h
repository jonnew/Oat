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
#include <string>

template <typename T> class SMServer {
    
public:
    SMServer(std::string block_name);
    SMServer(const SMServer& orig);
    virtual ~SMServer();
    
protected:
    
    T shared_object;

    std::string name;
    bool shared_write_object_created = false;
    boost::interprocess::managed_shared_memory shared_write_object;
    
    void createSharedObject(size_t bytes);
    
    //template<typename T>
    //void makeSharedObject(T data_object);

};

#endif	/* SMSERVER_H */

