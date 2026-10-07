#pragma once

// Process-wide receiver of /scene_repr/distance_field for the semantic
// collision plugin (P1.7.7).
//
// MoveIt creates and copies many CollisionEnv instances (one per planning
// scene diff), and its allocator hands them only the robot model, so the field
// subscription, the parameters and the TF lookup live in one lazily created
// node ("semantic_collision") spun on its own thread inside move_group.
// Parameters are read once at start-up; set them with a params file passed to
// the move_group process, under the key "semantic_collision".

#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "scout_piper_scene_repr/msg/semantic_distance_field.hpp"
#include "scout_piper_scene_repr/semantic_distance_field.hpp"

namespace scout_piper_scene_repr
{

struct SemanticCollisionConfig
{
  std::string field_topic{"/scene_repr/distance_field"};
  bool require_field{true};          // no usable field => every robot check reports collision
  bool unknown_is_occupied{true};    // unknown / stale voxels inside the grid are obstacles
  bool check_soft{true};             // apply the leaf penetration cap
  double max_field_age_s{5.0};       // message age (receiver clock - header.stamp); > publish period + export time
  double max_voxel_age_s{30.0};      // per-voxel observation age
  double sphere_padding_m{0.0};      // added to every robot sphere, on top of link padding
  double continuous_step{0.02};      // RobotState::distance per sub-check in continuous checks
  std::vector<std::string> ignore_links;   // never checked against the field
  std::string model_frame;           // "" = the robot model frame of the first env
};

struct FieldSnapshot
{
  std::shared_ptr<const SemanticDistanceField> field;
  Eigen::Isometry3d field_T_model{Eigen::Isometry3d::Identity()};   // model-frame point -> field frame
  bool transform_ok{false};
  double now{0.0};                   // receiver clock (s) when the snapshot was taken
  std::string reason;                // "" when usable, otherwise why not
};

class SemanticFieldListener
{
public:
  static SemanticFieldListener & instance();

  SemanticFieldListener(const SemanticFieldListener &) = delete;
  SemanticFieldListener & operator=(const SemanticFieldListener &) = delete;
  ~SemanticFieldListener();

  const SemanticCollisionConfig & config() const {return config_;}

  /// First caller fixes the planning frame used for the TF lookup (unless the
  /// model_frame parameter is set).
  void registerModelFrame(const std::string & frame);

  /// Latest field with its transform and the current time; ``reason`` says why
  /// it cannot be used (none yet, too old, missing TF, no ROS context).
  FieldSnapshot snapshot() const;

  rclcpp::Logger logger() const {return logger_;}

private:
  SemanticFieldListener();

  void onField(scout_piper_scene_repr::msg::SemanticDistanceField::ConstSharedPtr msg);
  void updateTransform();

  SemanticCollisionConfig config_;
  rclcpp::Logger logger_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::executors::SingleThreadedExecutor::SharedPtr executor_;
  std::thread spin_thread_;
  rclcpp::Subscription<scout_piper_scene_repr::msg::SemanticDistanceField>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr tf_timer_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  mutable std::mutex mutex_;
  std::shared_ptr<const SemanticDistanceField> field_;
  Eigen::Isometry3d field_T_model_{Eigen::Isometry3d::Identity()};
  bool transform_ok_{false};
  std::string model_frame_;
};

}  // namespace scout_piper_scene_repr
