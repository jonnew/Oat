//******************************************************************************
//* Copyright (c) Jon Newman (jpnewman at mit snail edu) 
//* All right reserved.
//* This file is part of the Simple Tracker project.
//* This is free software: you can redistribute it and/or modify
//* it under the terms of the GNU General Public License as published by
//* the Free Software Foundation, either version 3 of the License, or
//* (at your option) any later version.
//* This software is distributed in the hope that it will be useful,
//* but WITHOUT ANY WARRANTY; without even the implied warranty of
//* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//* GNU General Public License for more details.
//* You should have received a copy of the GNU General Public License
//* along with this source code.  If not, see <http://www.gnu.org/licenses/>.
//******************************************************************************

#ifndef SMCLIENT_H
#define	SMCLIENT_H

#include <string>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/sharable_lock.hpp>
#include <boost/interprocess/sync/interprocess_sharable_mutex.hpp>

template<class SyncType, class IOType>
class SMClient {
public:
    SMClient(std::string source_name);
    SMClient(const SMClient& orig);
    virtual ~SMClient();
    
    IOType get_value(void);

protected:
    
    
    
    SyncType* shared_object;

    std::string name, shmem_name, shobj_name;
    bool shared_object_found = false;
    boost::interprocess::managed_shared_memory cli_shared_memory;
    boost::interprocess::sharable_lock<boost::interprocess::interprocess_sharable_mutex> lock;

    boost::interprocess::sharable_lock<boost::interprocess::interprocess_sharable_mutex> makeLock();
    void findSharedObject(void);
   
};

#endif	/* SMCLIENT_H */

