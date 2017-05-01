//******************************************************************************
//* File:   IOFormat.h
//* Author: Jon Newman <jpnewman snail mit dot edu>
//*
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu)
//* All right reserved.
//* This file is part of the Oat project.
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

#ifndef OAT_IOFORMAT_H
#define OAT_IOFORMAT_H

#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unistd.h>

#define RST  "\x1B[0m"

#define FRED(x) "\x1B[31m" + x + RST
#define FGRN(x) "\x1B[32m" + x + RST
#define FYEL(x) "\x1B[33m" + x + RST
#define FBLU(x) "\x1B[34m" + x + RST
#define FMAG(x) "\x1B[35m" + x + RST
#define FCYN(x) "\x1B[36m" + x + RST
#define FWHT(x) "\x1B[37m" + x + RST

#define BOLD(x) "\x1B[1m" + x + RST
#define UNDL(x) "\x1B[4m" + x + RST

namespace oat {

/**
 * @brief List of terminal names known to support VT100 color escape sequences.
 * Taken from https://github.com/Kitware/CMake/tree/master/Source/kwsys
 */
static const char * vt100_term_names[] =
{
  "Eterm",
  "ansi",
  "color-xterm",
  "con132x25",
  "con132x30",
  "con132x43",
  "con132x60",
  "con80x25",
  "con80x28",
  "con80x30",
  "con80x43",
  "con80x50",
  "con80x60",
  "cons25",
  "console",
  "cygwin",
  "dtterm",
  "eterm-color",
  "gnome",
  "gnome-256color",
  "konsole",
  "konsole-256color",
  "kterm",
  "linux",
  "msys",
  "linux-c",
  "mach-color",
  "mlterm",
  "putty",
  "putty-256color",
  "rxvt",
  "rxvt-256color",
  "rxvt-cygwin",
  "rxvt-cygwin-native",
  "rxvt-unicode",
  "rxvt-unicode-256color",
  "screen",
  "screen-256color",
  "screen-256color-bce",
  "screen-bce",
  "screen-w",
  "screen.linux",
  "vt100",
  "xterm",
  "xterm-16color",
  "xterm-256color",
  "xterm-88color",
  "xterm-color",
  "xterm-debian",
  "xterm-termite",
  0
};

/**
 * @brief Detect whether a stream is displayed in a VT100-compatible terminal.
 * Taken from https://github.com/Kitware/CMake/tree/master/Source/kwsys
 *
 * @param stream Stream to check.
 * @param default_vt100 Assum VT100 is supported.
 * @param default_ttyi Assume that we are on a tty interface.
 *
 * @return 1 if VT100-compatible terminal is detected and 0 Otherwise.
 */
static int terminalStreamIsVT100(FILE * stream, int default_vt100=1, int default_tty=1) {

    // Force color according to http://bixense.com/clicolors/ convention.
    const char * clicolor_force = getenv("CLICOLOR_FORCE");
    if (clicolor_force && *clicolor_force && strcmp(clicolor_force, "0") != 0)
        return 1;

    // If running inside emacs the terminal is not VT100. Some emacs seem to
    // claim the TERM is xterm even though they do not support VT100 escapes.
    const char * emacs = getenv("EMACS");
    if(emacs && *emacs == 't')
        return 0;

    // Check for a valid terminal.
    if (!default_vt100) {

        const char ** t = 0;
        const char * term = getenv("TERM");
        if(term)
            for (t = vt100_term_names; *t && strcmp(term, *t) != 0; ++t) {}

        if (!(t && *t))
            return 0;
    }

    // Make sure the stream is a tty.
    (void)default_tty;
    return isatty(fileno(stream)) ? 1 : 0;
}

#define HAS_COLOR terminalStreamIsVT100(stdout)

inline std::string bold(const std::string &message) {

    return HAS_COLOR ? BOLD(message) : message;
}

inline std::string sourceText(const std::string &source_name) {

    return HAS_COLOR ? FGRN(source_name) : source_name;
}

inline std::string sinkText(const std::string& sink_name) {

    return HAS_COLOR ? FRED(sink_name) : sink_name;
}

inline std::string whoMessage(const std::string& source, const std::string& message) {

    return (HAS_COLOR ? BOLD(source) : source) + ": " + message;
}

inline std::string Warn(const std::string& message) {

    return HAS_COLOR ? FYEL(message) : message;
}

inline std::string Error(const std::string& message) {

    return HAS_COLOR ? FRED(message) : message;
}

inline std::string dbgMessage(const std::string& message) {

    return HAS_COLOR ? FMAG(message) : message;
}

inline std::string whoWarn(const std::string& source, const std::string& message) {

    return HAS_COLOR ?
        BOLD(source) + ": " + FYEL(message)
        :
        source +  ": " + message;
}

inline std::string whoError(const std::string& source, const std::string& message) {

    return HAS_COLOR ?
        BOLD(source) + ": " + FRED(message)
        :
        source +  ": " + message;
}


}      /* namespace oat */
#endif /* OAT_IOFORMAT_H */
