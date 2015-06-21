//******************************************************************************
//* File:   RegionFilter.cpp
//* Author: Jon Newman <jpnewman snail mit dot edu>
//
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu) 
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

#ifndef REGIONFILTER2D_H
#define	REGIONFILTER2D_H

#include <string>
#include <opencv2/opencv.hpp>

#include "PositionFilter.h"

 class RegionFilter2D : public PositionFilter {
    
public:
    RegionFilter2D(const std::string& position_source_name, const std::string& position_sink_name);
    ~RegionFilter2D();
    
    oat::Position2D filterPosition(oat::Position2D& position_in);
    
    void configure(const std::string& config_file, const std::string& config_key);
    
private:
    
    bool regions_configured;
    
    std::vector<std::string> region_ids;
    std::vector< std::vector<cv::Point> * > region_contours;

};

#endif	/* REGIONFILTER2D_H */

