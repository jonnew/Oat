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

#ifndef SMSERVER_H
#define	SMSERVER_H

#include <boost/interprocess/managed_shared_memory.hpp>
#include <string>

template <class SyncType>
class SMServer {
    
public:
    SMServer(std::string sink_name);
    SMServer(const SMServer& orig);
    virtual ~SMServer();

protected:
    
    SyncType* srv_shared_object;
    
    std::string srv_name, srv_shmem_name, srv_shobj_name, srv_mtx_name, srv_cond_name;
    bool srv_shared_write_object_created = false;
    boost::interprocess::managed_shared_memory srv_shared_memory;

    void createSharedObject(size_t bytes);
    void set_shared_object(SyncType val);
   
};

#endif	/* SMSERVER_H */

