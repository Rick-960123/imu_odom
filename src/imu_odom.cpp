#include "imu_odom/imu_odom.hpp"
#include "geometry_msgs/TwistStamped.h"
#include "tf/transform_datatypes.h"

ImuOdom::ImuOdom(ros::NodeHandle& nh) : nhi(nh)
{
  nh.param<std::string>("/robot_id", robot_id, "ZR1001");
  nh.param<std::string>("sub_topic", sub_topic, "/imu_calibrated");
  nh.param<std::string>("pub_topic", pub_topic, "/imu_odom");
  std::string pre = robot_id + "/imu0";
  nh.param<std::vector<double>>(pre + "/acc_sm", V_R_acc_mea, std::vector<double>());
  nh.param<std::vector<double>>(pre + "/gyr_sm", V_R_gyr_mea, std::vector<double>());

  imusub = nhi.subscribe(sub_topic, 32, &ImuOdom::ImuCallback, this);
  odompub = nhi.advertise<nav_msgs::Odometry>(pub_topic, 1);
  twist_pub = nhi.advertise<geometry_msgs::TwistStamped>("/imu_twist", 1);

  ROS_INFO("robot_id: %s", robot_id.c_str());
  ROS_INFO("sub_topic: %s", sub_topic.c_str());
  ROS_INFO("pub_topic: %s", pub_topic.c_str());

  //参数初始化
  odom.header.frame_id = "imu_odom";
  odom.child_frame_id = "base_link";
  imu_twist.header.frame_id = "base_link";

  point.pos = Eigen::Vector3d::Zero();
  point.rpy = Eigen::Vector3d::Zero();
  point.v = Eigen::Vector3d::Zero();
  point.w = Eigen::Vector3d::Zero();
  point.linear_x = 0.0;
  firstT = true;
}

void ImuOdom::ImuCallback(const sensor_msgs::Imu& imu_msg)
{
  // 获取IMU测量数据
  Eigen::Vector3d acc_mea(imu_msg.linear_acceleration.x, 0.0, 0.0);
  Eigen::Vector3d gyr_mea(0.0, 0.0, imu_msg.angular_velocity.z);

  if (firstT)
  {
    setGravity(acc_mea);
    last_time = imu_msg.header.stamp;
    firstT = false;
    return;
  }

  odom.header.seq = imu_msg.header.seq;
  odom.header.stamp = imu_msg.header.stamp;
  imu_twist.header.stamp = imu_msg.header.stamp;

  if (imu_msg.linear_acceleration.z < 1e-5 && imu_msg.linear_acceleration.z > -1e-5)
  {
    point.v[0] = 0.0;
    point.v[1] = 0.0;
    last_time = imu_msg.header.stamp;
    updateodom();
    return;
  }

  dt = (imu_msg.header.stamp - last_time).toSec();
  last_time = imu_msg.header.stamp;

  calcPosition(acc_mea);     //计算位置
  calcOrientation(gyr_mea);  //计算姿态
  updateodom();
}

void ImuOdom::setGravity(const Eigen::Vector3d& msg)
{
  gravity = msg;
}

void ImuOdom::calcOrientation(const Eigen::Vector3d& msg)
{
  point.w = msg;
  point.rpy += msg * dt;
}

void ImuOdom::calcPosition(const Eigen::Vector3d& msg)
{
  point.v[0] += msg[0] * cos(point.rpy[2]) * dt;
  point.v[1] += msg[0] * sin(point.rpy[2]) * dt;
  point.linear_x += msg[0] * dt;
  if (point.v.norm() > 0.6)
  {
    point.v = point.v / (point.v.norm() / 0.6);
  }
  point.pos = point.pos + dt * point.v;
}

void ImuOdom::updateodom()
{
  //位置
  odom.pose.pose.position.x = point.pos(0);
  odom.pose.pose.position.y = point.pos(1);
  odom.pose.pose.position.z = point.pos(2);

  //线速度
  odom.twist.twist.linear.x = point.v(0);
  odom.twist.twist.linear.y = point.v(1);
  odom.twist.twist.linear.z = point.v(2);

  //姿态
  odom.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(point.rpy(0), point.rpy(1), point.rpy(2));

  //角速度
  odom.twist.twist.angular.x = point.w(0);
  odom.twist.twist.angular.y = point.w(1);
  odom.twist.twist.angular.z = point.w(2);

  //发布里程计
  odompub.publish(odom);

  //发布twist
  imu_twist.twist.linear.x = point.linear_x;
  imu_twist.twist.linear.y = 0.0;
  imu_twist.twist.linear.z = 0.0;

  imu_twist.twist.angular.x = point.w(0);
  imu_twist.twist.angular.y = point.w(1);
  imu_twist.twist.angular.z = point.w(2);
  twist_pub.publish(imu_twist);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "imu_odom");
  ros::NodeHandle nh("~");
  ImuOdom *imu_to_odom = new ImuOdom(nh);
  ROS_INFO("start odom");
  ros::spin();
  return 0;
}
