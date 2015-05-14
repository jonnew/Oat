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
#include <boost/program_options.hpp>

namespace po = boost::program_options;

volatile sig_atomic_t done = 0;

void run(Viewer* viewer) {
 
    while (!done) {
        viewer->showImage();
    }
}

void printUsage(po::options_description options) {
    std::cout << "Usage: viewer [OPTIONS]\n"
    		  << "   or: viewer [CONFIGURATION] SOURCE\n"
    		  << "View the output of a SOURCE of type SMServer<SharedCVMatHeader>\n"
    		  << "Optionally, save the video stream to a file.\n"
    		  << options << "\n";
}

int main(int argc, char *argv[]) {

    // The image source to which the viewer will be attached
    std::string source;
    std::string file_name;
    bool append_date;
    
    try {

        po::options_description options("OPTIONS");
        options.add_options()
                ("help", "Produce help message.")
                ("version,v", "Print version information.")
                ;
        
        po::options_description hidden("CONFIGURATION");
        hidden.add_options()
                ("file,f", po::value<std::string>(&file_name),
                "The path to save the video stream to."
                ("date,d", po::value<std::string>(&append_date),
                "If specifed, YYYY-MM-DD-HH-mm-ss_ will be prepended to the filename.")
               ;

		po::options_description hidden("HIDDEN OPTIONS");
        hidden.add_options()
                ("source", po::value<std::string>(&source),
                "The name of the server that supplies images to view."
                "The server must be of type server<SharedCVMatHeader>\n")
                ;
        
        po::positional_options_description positional_options;
        positional_options.add("source", -1);
        
        po::options_description all_options("ALL OPTIONS");
        all_options.add(options).add(hidden);

        po::variables_map variable_map;
        po::store(po::command_line_parser(argc, argv)
                .options(all_options)
                .positional(positional_options)
                .run(),
                variable_map);
        po::notify(variable_map);

        // Use the parsed options
        if (variable_map.count("help")) {
            printUsage(options);
            return 0;
        }

        if (variable_map.count("version")) {
            std::cout << "Simple-Tracker Viewer, version 1.0\n"; //TODO: Cmake managed versioning
            std::cout << "Written by Jonathan P. Newman in the MWL@MIT.\n";
            std::cout << "Licensed under the GPL3.0.\n";
            return 0;
        }

        if (!variable_map.count("source")) {
            printUsage(options);
            std::cout << "Error: a SOURCE must be specified. Exiting.\n";
            return -1;
        }
        
        
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Exception of unknown type! " << std::endl;
    }
    
    // Make the viewer
    Viewer viewer(source);
    
    std::cout << "Viewer has begun listening to source \"" + source + "\".\n";
    std::cout << "COMMANDS:\n";
    std::cout << "  x: Exit.\n";
    
    // Two threads - one for user interaction, the other
    // for executing the processor
    boost::thread_group thread_group;
    thread_group.create_thread(boost::bind(&run, &viewer));
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
                std::cout << "Invalid command. Try again.\n";
                break;
        }
    }

    // TODO: If the server exits before the client, the client is blocked due
    // to the wait() call and therefore the done condition is never evaluated
    // and therefore the thread is never available for joining and therefore
    // this call hangs.
    thread_group.join_all();

    // Exit
    return 0;
}
