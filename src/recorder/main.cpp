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

#include "Recorder.h"

#include <algorithm>
#include <string>
#include <signal.h>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

volatile sig_atomic_t done = 0;

void run(Recorder* recorder) {

    while (!done) {
        recorder->writeStreams();
    }
}

void printUsage(po::options_description options) {
    std::cout << "Usage: record [OPTIONS]\n"
              << "   or: record [CONFIGURATION]\n"
              << "Record frame or position streams.\n\n"
              << options << "\n";
}

int main(int argc, char *argv[]) {

    std::vector<std::string> frame_sources;
    std::vector<std::string> position_sources;
    std::string file_name;
    std::string save_path;
    bool append_date = false;

    try {

        po::options_description options("OPTIONS");
        options.add_options()
                ("help", "Produce help message.")
                ("version,v", "Print version information.")
                ;

        po::options_description configuration("CONFIGURATION");
        configuration.add_options()
                ("filename,n", po::value<std::string>(&file_name),
                "The base file name to which to source name will be appended")
                ("folder,f", po::value<std::string>(&save_path),
                "The path to the folder to which the video stream and position information will be saved.")
                ("date,d",
                "If specified, YYYY-MM-DD-hh-mm-ss_ will be prepended to the filename.")
                ("positionsources,p", po::value< std::vector<std::string> >(),
                "The name of the server(s) that supply object position information."
                "The server(s) must be of type SMServer<Position>\n")
                ("imagesources,i", po::value< std::vector<std::string> >(),
                "The name of the server(s) that supplies images to save to video."
                "The server must be of type SMServer<SharedCVMatHeader>\n")
                ;

        po::options_description all_options("ALL OPTIONS");
        all_options.add(options).add(configuration);

        po::variables_map variable_map;
        po::store(po::command_line_parser(argc, argv)
                .options(all_options)
                .run(),
                variable_map);
        po::notify(variable_map);

        // Use the parsed options
        if (variable_map.count("help")) {
            printUsage(all_options);
            return 0;
        }

        if (variable_map.count("version")) {
            std::cout << "Simple-Tracker Decorator, version 1.0\n"; //TODO: Cmake managed versioning
            std::cout << "Written by Jonathan P. Newman in the MWL@MIT.\n";
            std::cout << "Licensed under the GPL3.0.\n";
            return 0;
        }

        if (!variable_map.count("positionsources") && !variable_map.count("imagesources")) {
            printUsage(all_options);
            std::cout << "Error: at least a single POSITION_SOURCE or FRAME_SOURCE must be specified. Exiting.\n";
            return -1;
        }

        if (!variable_map.count("folder") ) {
            save_path = ".";
            std::cout << "Warning: saving files to the current directory.\n";
        }
        
        if (!variable_map.count("filename") ) {
            file_name = "";
            std::cout << "Warning: no base filename was provided.\n";
        }

        // May contain imagesource and sink information!]
        if (variable_map.count("positionsources")) {
            position_sources = variable_map["positionsources"].as< std::vector<std::string> >();
            
            // Assert that all positions sources are unique. If not, remove duplicates, and issue warning.
            std::vector<std::string>::iterator it;
            it = std::unique (position_sources.begin(), position_sources.end());   
            if (it != position_sources.end()) {
                position_sources.resize(std::distance(position_sources.begin(),it)); 
                std::cout << "Warning: duplicate position sources have been removed.\n";
            }
            
        }
        
        if (variable_map.count("imagesources")) {
            frame_sources = variable_map["imagesources"].as< std::vector<std::string> >();
            
            // Assert that all positions sources are unique. If not, remove duplicates, and issue warning.
            std::vector<std::string>::iterator it;
            it = std::unique (frame_sources.begin(), frame_sources.end());   
            if (it != frame_sources.end()) {
                frame_sources.resize(std::distance(frame_sources.begin(),it)); 
                std::cout << "Warning: duplicate frame sources have been removed.\n";
            }
        }
        
        if (variable_map.count("date")) {
            append_date = true;
        } 

    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Exception of unknown type! " << std::endl;
    }

    std::cout << "Recorder started.\n";
    std::cout << "COMMANDS:\n";
    std::cout << "  x: Exit.\n";

    // Make the decorator
    Recorder recorder(position_sources, frame_sources, save_path, file_name, append_date);

    // Two threads - one for user interaction, the other
    // for processing
    boost::thread_group thread_group;
    thread_group.create_thread(boost::bind(&run, &recorder));
    sleep(1);

    // Start the user interface
    while (!done) {

        char user_input;
        std::cin >> user_input;

        switch (user_input) {
            case 'x':
            {
                done = true;
                break;
            }
            default:
                std::cout << "Invalid selection. Try again.\n";
                break;
        }
    }

    // Join processing and UI threads
    thread_group.join_all();

    std::cout << "Recorder is exiting." << std::endl;

    // Exit
    return 0;
}
