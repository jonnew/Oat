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

#include "BackgroundSubtractor.h"

#include <signal.h>

volatile sig_atomic_t done = 0;

void term(int) {
    done = 1;
}

int main(int argc, char *argv[]) {
    
    signal(SIGINT, term);

    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " SOURCE-NAME SINK-NAME " << std::endl;
        std::cout << "Background subtractor detector for cv::Mat data servers." << std::endl;
        std::cout << "Press \"B\" during acquisition to set background image to current frame." << std::endl;
        return 1;
    }
    
    const std::string source = static_cast<std::string> (argv[1]);
    const std::string sink = static_cast<std::string> (argv[2]);

    BackgroundSubtractor backsub(argv[1], argv[2]);

    std::cout << "Background subtractor has begun listening to source \"" + source + "\"." << std::endl;
    std::cout << "Background subtractor has begun steaming to sink \"" + sink + "\"." << std::endl;

    // Execute infinite, thread-safe loop with function calls governed by
    // underlying condition variable system.
    while (!done) {
        //if (getch()) {
            backsub.subtractBackground();
        //} else {
            //backsub.setBackgroundImageAndSubtract();
        //}
    }

    // Exit
    std::cout << "Background subtractor is exiting." << std::endl;
    return 0;
}


