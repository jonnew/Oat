//******************************************************************************
//* File:   CameraCalibrator.h
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

#ifndef OAT_CAMERACALIBRATOR_H
#define OAT_CAMERACALIBRATOR_H

#include <chrono>
#include <string>
#include <map>

#include <opencv2/core/mat.hpp>

#include "../../lib/utility/IOFormat.h"

#include "Calibrator.h"
#include "CalibratorVisitor.h"

namespace oat {

/**
 * Interactive camera calibration and distortion parameter generator.
 */
class CameraCalibrator : public Calibrator {

public:

    using Clock = std::chrono::high_resolution_clock;
    using Milliseconds = std::chrono::milliseconds;

    // Camera model to use for calibration
    enum class CameraModel
    {
        NA      = -1,  //!< Not applicable
        PINHOLE =  0,  //!< Pinhole camera model
        FISHEYE =  1   //!< Fisheye lens model
    };

    /**
     * Interactive camera calibrator. The corners of a chessboard pattern (of a
     * user-specified size and element length) are automatically detected in
     * the frame stream. These points are gathered, and upon the user's
     * request, used to generate a camera matrix and lens distortion
     * coefficient set. Two camera models are available: pinhole and fisheye.
     * @param frame_source_name imaging setup frame source name
     * @param model Camera model used to generate camera matrix and distortion coefficients.
     */
    CameraCalibrator(const std::string& frame_source_name,
                     const CameraModel& model,
                     cv::Size& chessboard_size,
                     double square_size_meters);

    /**
     * Configure calibration parameters.
     * @param config_file configuration file path
     * @param config_key configuration key
     */
    void configure(const std::string &config_file,
                   const std::string &config_key) override;

    // Accept visitors
    void accept(CalibratorVisitor* visitor) override;
    void accept(OutputVisitor* visitor, std::ostream& out) override;

    // Accessors
    CameraModel model() const { return model_; }
    bool calibration_valid() const { return calibration_valid_; }
    cv::Mat camera_matrix() const { return camera_matrix_; }
    cv::Mat distortion_coefficients() const { return distortion_coefficients_; }

protected:

    /**
     * Perform camera calibration routine.
     * @param frame current frame to use for running calibration
     */
    void calibrate(cv::Mat& frame) override;

private:

    // Interactive session mode
    enum class Mode
    {
        NORMAL = 0, //!< Top level commands available. Can enter/exit other modes.
        DETECT,     //!< Rig-detection mode. Used to populate chessboard corner data.
        UNDISTORT   //!< Undistort image using camera_matrix_ and distortion_coefficients_
    };

    Mode mode_ {Mode::NORMAL};

    // For writing mode on frames
    std::map<Mode, std::string> mode_msg_hash_;

    // Is camera calibration well-defined?
    CameraModel calibration_model_ {CameraModel::NA};
    bool calibration_valid_;
    cv::Mat camera_matrix_, distortion_coefficients_;
    double rms_error_;

    // Default estimation method
    CameraModel model_ {CameraModel::PINHOLE};

    // NXM black squares in the chessboard
    bool chessboard_detected_ {false};
    double square_size_meters_ {0.0254};
    cv::Size chessboard_size_; //!< Number of interior corners on chessboard

    // Frame dimensions
    cv::Size frame_size_;

    // Minimum time between chessboard corner detections
    Clock::time_point tick_, tock_;
    const Milliseconds min_detection_delay_ {1000};

    // Data used to camera calibration parameters
    std::vector<std::vector<cv::Point2f>> corners_;
    std::vector<cv::Point3f> corners_meters_;

    // Interactive session
    bool requireMode(const Mode&&, const Mode&&);
    void toggleMode(Mode);
    void detectChessboard(cv::Mat& frame);
    void undistortFrame(cv::Mat& frame);
    void decorateFrame(cv::Mat& frame);
    void printDataPoints(void);
    void clearDataPoints(void);
    void printCalibrationResults(std::ostream& out);
    void generateCalibrationParameters(void);
    int selectCameraModel(void);
    cv::Mat drawCorners(cv::Mat& frame, bool invert_colors);

    template<typename M>
    bool requireMode(M m) {

        bool correct_mode = mode_ == m;

        if (!correct_mode)
            std::cerr << oat::Error("Command unavailable in current mode.\n");

        return correct_mode;
    }

    template<typename M, typename... Args>
    bool requireMode(M m, Args&&... args) {
        return mode_ == m || requireMode(args...);
    }
};

}      /* namespace oat */
#endif /* OAT_CAMERACALIBRATOR_H */
