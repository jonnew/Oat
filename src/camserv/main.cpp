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


#include <unordered_map>
#include <signal.h>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/program_options.hpp>

#include "PGGigECam.h"
#include "WebCam.h"
#include "FileReader.h"

namespace po = boost::program_options;

volatile sig_atomic_t done = 0;
volatile bool running = true;

void term(int) {
    done = 1;
}

void run(Camera* camera) {

    while (!done) { // !done
        if (running) {
            camera->grabMat();
            camera->undistortMat();
            camera->serveMat();
        }
    }
    
    camera->stop();
}

void printUsage(po::options_description options) {
    std::cout << "Usage: camserv_ge [OPTIONS]\n";
    std::cout << "   or: camserv_ge TYPE SINK [CONFIGURATION]\n";
    std::cout << "Serve images captured by the camera to SINK\n\n";
    std::cout << "TYPE\n";
    std::cout << "  \'wcam\': Onboard or USB webcam.\n";
    std::cout << "  \'gige\': Point Grey GigE camera.\n";
    std::cout << "  \'file\': Stream video from file.\n\n";
    std::cout << options << "\n";
}

int main(int argc, char *argv[]) {

    // If ctrl-c is pressed, handle the signal with the term routine, which
    // will attempt to clean up the shared memory before exiting by calling
    // the object that is using shmem's destructor
    signal(SIGINT, term);

    std::string sink;
    std::string type;
    std::string video_file;
    std::string config_file;
    std::string config_key;
    bool config_used = false;
    po::options_description visible_options("VISIBLE OPTIONS");

    std::unordered_map<std::string, char> type_hash;
    type_hash["wcam"] = 'a';
    type_hash["gige"] = 'b';
    type_hash["file"] = 'c';

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
                ("video-file,f", po::value<std::string>(&video_file),
                "Path to video file if \'file\' is selected as the server TYPE.")
                ;

        po::options_description hidden("HIDDEN OPTIONS");
        hidden.add_options()
                ("type", po::value<std::string>(&type), "Camera TYPE.")
                ("sink", po::value<std::string>(&sink),
                "The name of the sink through which images collected by the camera will be served.\n")
                ;

        po::positional_options_description positional_options;
        positional_options.add("type", 1);
        positional_options.add("sink", 1);

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

        if (!variable_map.count("type")) {
            printUsage(visible_options);
            std::cout << "Error: a TYPE must be specified. Exiting.\n";
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

        if (type.compare("file") == 0 && !variable_map.count("video-file")) {
            printUsage(visible_options);
            std::cout << "Error: when TYPE=file, a video-file path must be specified. Exiting.\n";
            return -1;
        }


    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Exception of unknown type!" << std::endl;
    }

    // Create the specified TYPE of detector
    Camera* camera;
    switch (type_hash[type]) {
        case 'a':
        {
            camera = new WebCam(sink);
            break;
        }
        case 'b':
        {
            camera = new PGGigECam(sink);
            break;
        }
        case 'c':
        {
            camera = new FileReader(video_file, sink);
            break;
        }
        default:
        {
            printUsage(visible_options);
            std::cout << "Error: invalid TYPE specified. Exiting.\n";
            return -1;
        }
    }

    if (config_used)
        camera->configure(config_file, config_key);
    else
        camera->configure();

    std::cout << "Camera named \"" + sink + "\" has started.\n";
    std::cout << "COMMANDS:\n";
    std::cout << "  p: Pause/unpause.\n";
    std::cout << "  x: Exit.\n";

    // Two threads - one for user interaction, the other
    // for executing the processor
    boost::thread_group thread_group;
    thread_group.create_thread(boost::bind(&run, camera));
    sleep(1);

    while (!done) {
        
        char user_input;
        std::cin >> user_input;

        switch (user_input) {

            case 'p':
            {
                running = !running;
                break;
            }
            case 'x':
            {
                done = true;
                camera->stop(); // TODO: Necessary??
                break;
            }
            default:
                std::cout << "Invalid command. Try again.\n";
                break;
        }
    }

    // Exit gracefully and ensure all shared resources are cleaned up
    thread_group.join_all();
    
    // Free heap memory allocated to camera 
    delete camera;

    std::cout << "Camera named \"" + sink + "\" is exiting." << std::endl;

    // Exit
    return 0;
}
