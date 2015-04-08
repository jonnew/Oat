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

#ifndef MATCLIENT_H
#define	MATCLIENT_H

#include "SharedMat.h"
#include <string>

class MatClient {
public:
    MatClient(const std::string server_name);
    MatClient(const MatClient& orig);
    virtual ~MatClient();
    
    std::string get_cli_name(void) { return cli_name; }
    
protected:
    void findSharedMat(void);
    cv::Mat get_shared_mat(void);
    std::string cli_name;
    shmem::SharedMatHeader* cli_shared_mat_header;
    bool cli_shared_mat_created = false;

private:
    
    cv::Mat mat;

    void* shared_mat_data_ptr;
    int data_size; // Size of raw mat data in bytes
    
    std::string shmem_name, shobj_name;
    boost::interprocess::managed_shared_memory shared_memory;

};

#endif	/* MATCLIENT_H */

