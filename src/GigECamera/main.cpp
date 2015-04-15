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

#include "CameraControl.h"

#include <unordered_map>
#include <signal.h>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

volatile sig_atomic_t done = 0;
bool running = true;

void term(int) {
    done = 1;
}

void run(CameraControl* cc, std::string name) {

    std::cout << "Camera named \"" + name + "\" has started.\n";

    while (!done) { // !done
        if (running) {
            cc->serveMat();
        }
    }
 
    std::cout << "Camera named \"" + name + "\" is exiting." << std::endl;
}

void printUsage(po::options_description options) {
    std::cout << "Usage: camserv_ge [OPTIONS]\n";
    std::cout << "   or: camserv_ge CAMERA SINK [CONFIGURATION]\n";
    std::cout << "Serve images captured by the camera to SINK\n";
    std::cout <<  "\n";
    std::cout <<  "CAMERA\n";
    std::cout <<  "  \'wcam\': Onboard or USB webcam.\n";
    std::cout <<  "  \'gige\': Point Grey GigE camera.\n\n";
    std::cout << options << "\n";
}

int main(int argc, char *argv[]) {

    // If ctrl-c is pressed, handle the signal with the term routine, which
    // will attempt to clean up the shared memory before exiting by calling
    // the object that is using shmem's destructor
    signal(SIGINT, term);

    std::string sink;
    std::string camera_code;
    std::string config_file;
    std::string config_key;
    bool config_used = false;

    std::unordered_map<std::string, int> camera_hash;
    camera_hash.insert({"wcam", 0});
    camera_hash.insert({"gige", 1});

    try {

        po::options_description options("OPTIONS");
        options.add_options()
                ("help", "Produce help message.")
                ("version,v", "Print version information.")
                ;

        po::options_description config("CONFIGURATION");
        config.add_options()
                ("config-file,c", po::value<std::string>(&config_file), "Configuration file.")
                ("config-key,k", po::value<std::string>(&config_key), "Configuration key.")
                ;

        po::options_description hidden("HIDDEN OPTIONS");
        hidden.add_options()
		("camera", po::value<std::string>(&camera_code), "Camera code.")
                ("sink", po::value<std::string>(&sink),
                "The name of the sink through which images collected by the camera will be served.\n")
                ;

        po::positional_options_description positional_options;
        positional_options.add("camera", 1);
        positional_options.add("sink", 2);

        po::options_description visible_options("VISIBLE OPTIONS");
        visible_options.add(options).add(config);

        po::options_description all_options("ALL OPTIONS");
        all_options.add(options).add(config).add(hidden);

        po::variables_map variable_map;
        po::store(po::command_line_parser(argc, argv)
                .options(all_options)
                .positional(positional_options)
                .run(),
                variable_map);
        po::notify(variable_map);

        // Use the parsed options
        if (variable_map.count("help")) {
            printUsage(visible_options);
            return 0;
        }

        if (variable_map.count("version")) {
            std::cout << "Simple-Tracker GigECamera Server version 1.0\n"; //TODO: Cmake managed versioning
            std::cout << "Written by Jonathan P. Newman in the MWL@MIT.\n";
            std::cout << "Licensed under the GPL3.0.\n";
            return 0;
        }

        if (!variable_map.count("camera")) {
            printUsage(visible_options);
            std::cout << "Error: a CAMERA must be specified. Exiting.\n";
            return -1;
        } 

        if (!variable_map.count("sink")) {
            printUsage(visible_options);
            std::cout << "Error: a SINK must be specified. Exiting.\n";
            return -1;
        } 
       
        if ((variable_map.count("config-file") && !variable_map.count("config-key")) ||
                (!variable_map.count("config-file") && variable_map.count("config-key"))) {
            printUsage(visible_options);
            std::cout << "Error: config file must be supplied with a corresponding config-key. Exiting.\n";
           return -1;
        } else if (variable_map.count("config-file")) {
            config_used = true;
        }
        

    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Exception of unknown type!" << std::endl;
    }
    
    CameraControl cc(sink); // TODO: CameraControl should be a class template that allows different camera types with common methods 
    if (config_used)
        cc.configure(config_file, config_key);
    else
        cc.configure();
    
    // Two threads - one for user interaction, the other
    // for executing the processor
    boost::thread_group thread_group;
    thread_group.create_thread(boost::bind(&run, &cc, sink));
    sleep(1);
    
    while (!done) {
        int user_input;
        std::cout << "Select an action:\n";
        std::cout << " [1]: Pause/unpause\n";
        std::cout << " [2]: Exit\n";
        std::cout << ">> ";

        std::cin >> user_input;

        switch (user_input) {

            case 1:
            {
                running = !running;
                if (running)
                    std::cout << " Resumed...\n";
                else
                    std::cout << " Paused.\n";
                break;
            }

            case 2:
            {
                done = true;
                break;
            }
            default:
                std::cout << "Invalid selection. Try again.\n";
                break;
        }
    }

    // TODO: Exit gracefully and ensure all shared resources are cleaned up!
    thread_group.join_all();
    
    // Exit
    return 0;
}
