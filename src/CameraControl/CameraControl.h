#ifndef CameraControl_H
#define CameraControl_H

#include "FlyCapture2.h"
#include <opencv2/core/core.hpp>
#include "SimpleTrackerConfig.h"

using namespace std;
using namespace FlyCapture2;

class CameraControl {

	public:
		CameraControl(void);

		// For establishing connection
		int print_bus_info(void);
		int set_camera_index(unsigned int requested_idx);
		int connect_to_camera(void);

		// Once connected
		int turn_camera_on(void);
		int setup_trigger(int source, int polarity);
		cv::Mat grab_image(void);

		// int turn_camera_off(void);
		void get_camera_info(void);
		int print_camera_info(void);

	private:
		bool aquisition_started;
		unsigned int num_cameras, index;
		Camera camera;
		CameraInfo cam_info;
		TriggerMode trig_mode;
		TriggerModeInfo trig_mode_info;
		PGRGuid guid;
		BusManager busMgr;


		int find_num_cameras(void); 
		void print_error(Error error);
		bool poll_for_trigger_ready(void);

};

#endif //CameraConfig_H
