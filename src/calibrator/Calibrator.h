//******************************************************************************
//* File:   Calibrator.h
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
//****************************************************************************

#ifndef OAT_CALIBRATOR_H
#define OAT_CALIBRATOR_H

#include <string>
#include <iosfwd>
#include <opencv2/core/mat.hpp>

#include "../../lib/datatypes/Frame.h"
#include "../../lib/shmemdf/Source.h"

namespace oat {

// Forward decl.
class CalibratorVisitor;
class OutputVisitor;
class SharedFrameHeader;

/**
 * Abstract calibrator.
 * All concrete calibrator types implement this ABC.
 */
class Calibrator {

public:

    /**
     * Abstract calibrator.
     * All concrete calibrator types implement this ABC.
     * @param frame_source_address Frame SOURCE address
     */
    Calibrator(const std::string &frame_source_address,
               const std::string &calibration_key);

    virtual ~Calibrator() {};

    /**
     * Calibrator SOURCE must be able to connect to a NODE from
     * which to receive frames.
     */
    virtual void connectToNode(void);

    /**
     * Run the calibration routine on the frame SOURCE.
     * @return True if SOURCE signals EOF
     */
    virtual bool process(void);

    /**
     * Configure calibration parameters.
     * @param config_file configuration file path
     * @param config_key configuration key
     */
    virtual void configure(const std::string &config_file,
                           const std::string &config_key) = 0;

    /**
     * Create the calibration file path using a specified path.
     * @param save_path the path to save configuration data. If this path is a
     * folder, the calibration file will default to calibraiton.toml.
     * @return True if the specified calibration file already exists.
     */
    virtual bool generateSavePath(const std::string& save_path);

    // Accept functions for visitors
    virtual void accept(CalibratorVisitor* visitor) = 0;
    virtual void accept(OutputVisitor* visitor, std::ostream& out) = 0;

    // Accessors
    const std::string & name() const { return name_; }
    const std::string & calibration_save_path() const {
        return calibration_save_path_;
    }

    void set_calibration_key(const std::string& value) {calibration_key_ = value; }

protected:

    /** Perform calibration routine.
     * @param frame frame to use for generating calibration parameters
     */
    virtual void calibrate(cv::Mat& frame) = 0;

    std::string calibration_key_;       //!< Key name of calibration table entry
    std::string calibration_save_path_; //!< Calibration parameter save path

private:

    std::string name_;                      //!< Calibrator name
    oat::Frame internal_frame_;             //!< Current frame provided by SOURCE
    std::string frame_source_address_;      //!< Frame source address
    oat::NodeState node_state_;             //!< Frame source node state
    oat::Source<SharedFrameHeader> frame_source_; //!< The calibrator frame SOURCE
};

}      /* namespace oat */
#endif /* OAT_CALIBRATOR_H */
