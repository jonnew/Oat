//******************************************************************************
//* File:   OutputVisitor.h
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

#ifndef OAT_OUTPUTVISITOR_H
#define	OAT_OUTPUTVISITOR_H

#include <iosfwd>

namespace oat {

// Foward decl.
class CameraCalibrator;
class HomographyGenerator;

/**
 *
 */
class OutputVisitor {

public:
    // Visitor with supplied output stream (e.g. for printing usage info)
    virtual void visit(CameraCalibrator *cc, std::ostream &out) = 0;
    virtual void visit(HomographyGenerator *hg, std::ostream &out) = 0;
};

}      /* namespace oat */
#endif /* OAT_OUTPUTVISITOR_H */

