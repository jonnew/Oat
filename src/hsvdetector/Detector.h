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

#ifndef DETECTOR_H
#define	DETECTOR_H

#include <string>

#include "../../lib/shmem/Position2D.h"
#include "../../lib/shmem/MatClient.h"
#include "../../lib/shmem/SMServer.h"

class Detector {
public:
    
    virtual void findObject(void) = 0;
    virtual void servePosition(void) = 0;
    
protected:
    
    // Method for sifting a threshold image to find objects
    virtual void siftBlobs(void) = 0;
    
    // Detectors must allow manual tuning
    bool tuning_on;
    std::string slider_title;
    virtual void createSliders(void) = 0;
    
    // The detected object position
    shmem::Position2D object_position;
    
    // The image source (Client side)
    MatClient image_source;
    
    // The detected object position destination (Server side)
    shmem::SMServer<shmem::Position2D> position_sink;
    
};

#endif	/* DETECTOR_H */

