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
bool running = true;

void term(int) {
	done = 1;
}

void run(std::string source) {

	Viewer viewer(source);
	std::cout << "Viewer has begun listening to source \"" + source + "\"." << std::endl;

	while (!done) { 
		if (running) {
			viewer.showImage();
		}
	}

	std::cout << "Viewer listening to source \"" + source + "\" is exiting." << std::endl;
}

int main(int argc, char *argv[]) {

	// If Ctrl-C is pressed, handle the signal with the term routine
	signal(SIGINT, term);

	// The image source to view
	const std::string source = static_cast<std::string> (argv[1]);

	try {
		int opt;

		po::options_description basic_options("Basic options");
		basic_options.add_options()
			("version,v", "print version number")
			("help", "produce help message")
			("source,s", po::value<string>(&source), 
			 " The name of the server that supplies images to view.\n"
			 " The server must be of type server<SharedCVMatHeader>\n")
			;

		po::positional_options_description positional_options;
		positional_options.add("source", -1);

		po::variables_map variable_map;
		po::store(po::command_line_parser(argc, argv)
				.options(basic_options)
				.positional(positional_options)
				.run(), 
				variable_map);
		po::notify(variable_map);

		// Use the parsed options
		if (variable_map.count("version")) {
			std::cout << "Simple-Tracker Viewer, version 1.0" << std::endl; //TODO: Cmake managed versioning
			exit(EXIT_SUCCESS);
		}

		if (variable_map.count("help")) {
			std::cout << basic_options << std::endl;
			exit(EXIT_SUCCESS);
		}

		if (!variable_map.count("source")) {
			std::cout << "An image SOURCE must be specified. Exiting." << std::endl;
			exit(EXIT_FAILURE);
		}
	} 
	catch (exception& e) {
		cerr << "error: " << e.what() << "\n";
		return 1;
	}
	catch (...) {
		cerr << "Exception of unknown type!\n";
	}

	// Two threads - one for user interaction, the other
	// for executing the viewer
	boost::thread_group thread_group;
	thread_group.create_thread(boost::bind(&run, source));

	while (!done) {

		int user_input;
		std::cout << std::endl;
		std::cout << "Select an action:" << std::endl;
		std::cout << " [1]: Pause/unpause viewer " << std::endl;
		std::cout << " [2]: Exit viewer " << std::endl;
		std::coud << ">> ";

		std::cin >> user_input;

		switch (user_input) {
			case 1: {
						running = !running;
						break;
					}
			case 2: {
						done = true;
						break;
					}
			default:
					std::cout << "Invalid selection. Try again." << std::endl;
					break;
		}
	}

	// TODO: Exit gracefully and ensure all shared resources are cleaned up. This might already
	// be functional, but I'm not sure...
	thread_group.join_all();

	// Exit
	exit(EXIT_SUCCESS);
}
