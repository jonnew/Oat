//******************************************************************************
//* Copyright (c) Jon Newman (jpnewman at mit snail edu) 
//* All right reserved.
//* This file is part of the Oat project.
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

#ifndef MATSERVER_H
#define	MATSERVER_H


#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <opencv2/core/mat.hpp>

#include "SharedMemoryManager.h"
#include "SharedCVMatHeader.h"

namespace oat {

    // TODO: Find a why to integrate this with the must much general purpose SMServer
    class MatServer {
    public:
        MatServer(const std::string& sink_name);
        MatServer(const MatServer& orig);
        virtual ~MatServer();

        void createSharedMat(void);
        void pushMat(const cv::Mat& mat, const uint32_t& sample_number);
        void setSharedServerState(oat::ServerRunState state);
      
        // Accessors 
        std::string get_name(void) const { return name; }

    private:

        // Name of this server
        std::string name;

        // Shared object control
        oat::SharedCVMatHeader* shared_mat_header;
        oat::SharedMemoryManager* shared_mem_manager;
        bool* eof_signal;
        bool shared_object_created;
        bool mat_header_constructed;
        int data_size; // Size of raw mat data in bytes

        const std::string shmem_name, shobj_name, shmgr_name;
        boost::interprocess::managed_shared_memory shared_memory;

        /**
         * Auto post to this servers own semaphore.wait() to allow threads to unblock
         * in order to ensure proper object destruction.
         * @param mat
         */
        void notifySelf(void);

    };

}

#endif	/* MATSERVER_H */

