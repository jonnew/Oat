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

#ifndef MEANPOSITION2D_H
#define	MEANPOSITION2D_H

#include "PositionCombiner.h"

class MeanPosition2D : public PositionCombiner {
public:
    
    MeanPosition2D(std::vector<std::string> position_source_names, std::string sink_name);
    
    void configure(const std::string& config_file, const std::string& config_key);
private:

    oat::Position2D combinePositions(const std::vector<oat::Position2D*>& sources);
    
    bool generate_heading;
    int heading_anchor_idx;

};

#endif	/* MEANPOSITION2D_H */

