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

#include "Viewer.h"

#include <string>
#include <signal.h>
#include <boost/thread.hpp>
#include <boost/bind.hpp>

volatile sig_atomic_t done = 0;
bool running = true;

void term(int) {
    done = 1;
}

void run(std::string source) {

    Viewer viewer(source);
    std::cout << "A viewer has begun listening to source \"" + source + "\"." << std::endl;

    while (!done) { // !done
        if (running) {
            viewer.showImage();
        }
    }
}

int main(int argc, char *argv[]) {

    signal(SIGINT, term);

    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " SOURCE-NAME" << std::endl;
        std::cout << "Viewer for cv::Mat data servers" << std::endl;
        return 1;
    }

    const std::string source = static_cast<std::string> (argv[1]);

    boost::thread_group thread_group;
    thread_group.create_thread(boost::bind(&run, source));

    // TODO: Standard startup dialog (common to all simple-tracker programs, perhaps)

    while (!done) {
        int user_input;
        std::cout << std::endl;
        std::cout << "Select an action:" << std::endl;
        std::cout << "   [1]: Pause/unpause viewer " << std::endl;
        std::cout << "   [2]: Exit viewer " << std::endl;
        std::cin >> user_input;

        switch (user_input) {

            case 1:
            {
                running = !running;
                break;
            }

            case 2:
            {
                done = true;
                break;
            }
            default:

                std::cout << "Invalid selection. Try again." << std::endl;
                break;
        }
    }

    // TODO: Exit gracefully and ensure all shared resources are cleaned up!
    thread_group.join_all();

    // Exit
    std::cout << "Viewer listening to source \"" + source + "\" is exiting." << std::endl;
    return 0;
}
