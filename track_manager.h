#ifndef TRACK_MANAGER_ROS_H
#define TRACK_MANAGER_ROS_H

typedef struct {
	double x; // meters
	double y;
	double z;
	double roll; // radians
	double pitch;
	double yaw;
} pose_t;

typedef struct {
	float x; // meters
	float y;
	float z;
	float intensity;
} point_t;

#endif //TRACK_MANAGER_H
