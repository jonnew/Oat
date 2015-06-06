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

#ifndef FILEREADER_H
#define	FILEREADER_H

#include <string>
#include <opencv2/opencv.hpp>

#include "Camera.h"

class FileReader : public Camera {
public:
    
    FileReader(std::string file_name_in, 
               std::string image_sink_name, 
               const double& frames_per_second = 30);
    
    // Implement Camera interface
    void configure(void); 
    void configure(std::string config_file, std::string key);
    void grabMat(void);
    void serveMat(void);
    
private:
    
    std::string file_name;
    int frame_period_in_us;
    double frame_rate_in_hz;
    void calculateFramePeriod(void);
    
    // File read
    cv::VideoCapture file_reader;
    
    // Should the image be cropped
    bool use_roi;
};

#endif	/* FILEREADER_H */

