//******************************************************************************
//* File:   FileReader.cpp
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

#include "FileReader.h"

#include <thread>

#include <cpptoml.h>

#include "../../lib/utility/IOFormat.h"
#include "../../lib/utility/TOMLSanitize.h"

namespace oat {

FileReader::FileReader(const std::string &sink_address)
: FrameServer(sink_address)
{
    // Initialize time
    tick_ = clock_.now();
}

po::options_description FileReader::options() const
{
    // Update CLI options
    po::options_description local_opts;
    local_opts.add_options()
        ("video-file,f", po::value<std::string>(),
         "Path to video file to serve frames from.")
        ("fps,r", po::value<double>(),
         "Frames to serve per second.")
        ("roi", po::value<std::string>(),
         "Four element array of unsigned ints, [x0,y0,width,height],"
         "defining a rectangular region of interest. Origin"
         "is upper left corner. ROI must fit within acquired"
         "frame size. Defaults to full video size.")
        ;

    return local_opts;
}

void FileReader::applyConfiguration(const po::variables_map &vm,
                                    const config::OptionTable &config_table)
{
    // Video file
    std::string file_name;
    oat::config::getValue(vm, config_table, "video-file", file_name, true);
    file_reader_.open(file_name);

    // Frame rate
    if (oat::config::getNumericValue(vm, config_table, "fps", frames_per_second_, 0.0))
        calculateFramePeriod();

    // ROI
    std::vector<size_t> roi;
    if (oat::config::getArray<size_t, 4>(vm, config_table, "roi", roi)) {

        use_roi_ = true;
        region_of_interest_.x      = roi[0];
        region_of_interest_.y      = roi[1];
        region_of_interest_.width  = roi[2];
        region_of_interest_.height = roi[3];
    }
}

bool FileReader::connectToNode()
{
    cv::Mat example_frame;
    file_reader_ >> example_frame;

    if (use_roi_)
        example_frame = example_frame(region_of_interest_);

    frame_sink_.bind(frame_sink_address_,
            example_frame.total() * example_frame.elemSize());

    shared_frame_ = frame_sink_.retrieve(
            example_frame.rows, example_frame.cols, example_frame.type(), PIX_BGR);

    // Reset the video to the start
    file_reader_.set(cv::CAP_PROP_POS_AVI_RATIO, 0);

    // Put the sample rate in the shared frame
    shared_frame_.set_rate_hz(1.0 / frame_period_in_sec_.count());

    return true;
}

int FileReader::process()
{
    cv::Mat frame;
    if (!file_reader_.read(frame))
        return 1;

    if (use_roi_ )
        frame = frame(region_of_interest_);

    // START CRITICAL SECTION //
    ////////////////////////////

    // Wait for sources to read
    frame_sink_.wait();

    frame.copyTo(shared_frame_);
    shared_frame_.incrementSampleCount();

    // Tell sources there is new data
    frame_sink_.post();

    ////////////////////////////
    //  END CRITICAL SECTION  //

    std::this_thread::sleep_for(frame_period_in_sec_ - (clock_.now() - tick_));
    tick_ = clock_.now();

    return 0;
}

void FileReader::calculateFramePeriod()
{
    // Copy assignment provides automatic unit conversion
    std::chrono::duration<double> frame_period {1.0 / frames_per_second_};
    frame_period_in_sec_ = frame_period;
}

} /* namespace oat */
