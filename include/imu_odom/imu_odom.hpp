#ifndef IMU_TO_ODOM_IMU_TO_ODOM_HPP
#define IMU_TO_ODOM_IMU_TO_ODOM_HPP
// ROS includes
#include "geometry_msgs/TwistStamped.h"
#include "ros/ros.h"
#include "ros/time.h"
#include <sensor_msgs/Imu.h>
#include <Eigen/Dense>
#include <cmath>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <string>
#include <vector>
#include <tf/tf.h>
#include <iomanip>
struct Piont
{
  Eigen::Vector3d pos;  //位置
  Eigen::Vector3d rpy;  //姿态
  Eigen::Vector3d w;    //角速度
  Eigen::Vector3d v;    //线速度
  double linear_x;
};

// imu处理
class ImuOdom
{
private:
  ros::NodeHandle nhi;
  ros::Subscriber imusub;
  ros::Publisher odompub, twist_pub;
  nav_msgs::Odometry odom;
  geometry_msgs::TwistStamped imu_twist;
  ros::Time last_time;
  Piont point;
  Eigen::Vector3d gravity;
  double dt;
  bool firstT;

public:
  //! Constructor.
  ImuOdom(ros::NodeHandle& nh);
  //! Destructor.
  ~ImuOdom();
  void ImuCallback(const sensor_msgs::Imu& imu_msg);
  void setGravity(const Eigen::Vector3d& msg);
  void calcPosition(const Eigen::Vector3d& msg);
  void calcOrientation(const Eigen::Vector3d& msg);
  void updateodom();

  std::string robot_id;
  std::string imu_name;
  std::string sub_topic;
  std::string pub_topic;
  std::vector<double> V_R_acc_mea, V_R_gyr_mea;
  Eigen::Matrix3d R_acc_mea, R_gyr_mea;
  Eigen::Vector3d acc_noise, acc_bias, gyr_noise, gyr_bias;
};
#endif