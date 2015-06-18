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

#ifndef BACKGROUNDSUBTRACTOR_H
#define	BACKGROUNDSUBTRACTOR_H

#include "FrameFilter.h"

class BackgroundSubtractor : public FrameFilter {
public:

    BackgroundSubtractor(const std::string& source_name, const std::string& sink_name);

    void setBackgroundImage(const cv::Mat&);
    void configure(const std::string& config_file, const std::string& config_key);
    cv::Mat filter(cv::Mat& frame);
    
private:

    // The background image used for subtraction
    bool background_set = false;
    cv::Mat background_img;

};

#endif	/* BACKGROUNDSUBTRACTOR_H */

