//******************************************************************************
//* File:   UsagePrinter.h
//* Copyright (c) Jon Newman (jpnewman snail mit dot edu)
//
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

#ifndef OAT_USAGEPRINTER_H
#define OAT_USAGEPRINTER_H

#include <iosfwd>

#include "OutputVisitor.h"

namespace oat {

// Forward declarations
class CameraCalibrator;
class HomographyGenerator;

/**
 *
 */
class UsagePrinter : public OutputVisitor {

public:

    /**
     * Print interactive usage information for the CameraCalibrator.
     * @param out output stream to print usage information to.
     */
    void visit(CameraCalibrator* cc, std::ostream& out) override;

    /**
     * Print interactive usage information for the HomographyGenerator.
     * @param out output stream to print usage information to.
     */
    void visit(HomographyGenerator* hg, std::ostream& out) override;

};

}      /* namespace oat */
#endif /* OAT_USAGEPRINTER_H */

