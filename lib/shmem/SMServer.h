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
    
    void createSharedObject(void);
    void set_value(SyncType value);
    
    
protected:
    
    SyncType* shared_object;
    
    std::string name;
    std::string shmem_name, shobj_name;
    boost::interprocess::managed_shared_memory shared_memory;

    void createSharedObject(size_t bytes);

};

#endif	/* SMSERVER_H */

