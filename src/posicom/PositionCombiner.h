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

#ifndef POSITIONCOMBINER_H
#define	POSITIONCOMBINER_H

#include <string>

#include "../../lib/shmem/SMServer.h"
#include "../../lib/shmem/SMClient.h"
#include "../../lib/shmem/Position.h"

class PositionCombiner {
    
    public:
        
        PositionCombiner(std::string antierior_source, std::string posterior_source, std::string sink);
                
        void calculateGeometricMean(void);
        void serveCombinedPosition(void);
        void stop(void);
        
        std::string get_name(void) { return name; }
        
    private:
        
        std::string name;
        
        // Anterior and posterior position measures
        shmem::SMClient<shmem::Position> anterior_source;
        shmem::SMClient<shmem::Position> posterior_source;
        
        // Processed position server
        shmem::SMServer<shmem::Position> position_sink;
        
        shmem::Position processed_position;  
};

#endif	// POSITIONCOMBINER_H

