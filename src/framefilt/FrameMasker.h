//******************************************************************************
//* File:   FrameMasker.h
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

#ifndef FRAMEMASKER_H
#define	FRAMEMASKER_H

#include "FrameFilter.h"

class FrameMasker : public FrameFilter {
public:
    
    FrameMasker(const std::string& source_name, 
                const std::string& sink_name, 
                bool invert_mask=false);
    
    void filterAndServe(void);
    void configure(const std::string& config_file, const std::string& config_key);
    
    // Accessors
    void set_invert_mask(bool value) { invert_mask = value; }
    
private:
    
    // Should the mask be inverted?
    // This value can be accessed from the UI thread
    std::atomic<bool> invert_mask;
    
    // Do we have a mask to work with
    bool mask_set = false;
    
    // Mask frames with an arbitrary ROI
    cv::Mat roi_mask;

};

#endif	/* FRAMEMASKER_H */

