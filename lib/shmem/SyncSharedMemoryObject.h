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

#ifndef SYNCSHAREDMEMORYOBJECT_H
#define	SYNCSHAREDMEMORYOBJECT_H

#include <boost/interprocess/sync/interprocess_sharable_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition_any.hpp>


namespace shmem {
    
    class SyncSharedMemoryObject {
    public:
        
        bool ready = false;
        boost::interprocess::interprocess_sharable_mutex mutex;
        boost::interprocess::interprocess_condition_any new_data_condition;
        
    };
}


#endif	/* SYNCSHAREDMEMORYOBJECT_H */
