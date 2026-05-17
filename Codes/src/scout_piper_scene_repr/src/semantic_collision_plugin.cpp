// Semantic collision detector — Phase 1 skeleton.
//
// This file registers a MoveIt 2 collision detector with pluginlib. The plugin
// is wired through `collision_detector: "Semantic"` in the move_group config.
//
// Skeleton policy: every check delegates to a Hybrid-FCL self-collision path
// while the per-class ESDF queries are stubbed. See P1.4 in PROGRESS.md.

#include "scout_piper_scene_repr/semantic_collision_plugin.hpp"

#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>

namespace scout_piper_scene_repr
{

const std::string SemanticCollisionDetectorAllocator::NAME = "Semantic";

CollisionEnvSemantic::CollisionEnvSemantic(
  const moveit::core::RobotModelConstPtr & robot_model)
: collision_detection::CollisionEnv(robot_model)
, policies_{}
{
  RCLCPP_INFO(
    rclcpp::get_logger("scout_piper_scene_repr.semantic_collision_plugin"),
    "Created CollisionEnvSemantic (single-arg ctor); policies empty — load via setPolicies().");
}

CollisionEnvSemantic::CollisionEnvSemantic(
  const moveit::core::RobotModelConstPtr & robot_model,
  const collision_detection::WorldPtr & world)
: collision_detection::CollisionEnv(robot_model, world)
, policies_{}
{
}

void CollisionEnvSemantic::setPolicies(const ClassPolicyMap & policies)
{
  policies_ = policies;
  RCLCPP_INFO(
    rclcpp::get_logger("scout_piper_scene_repr.semantic_collision_plugin"),
    "Loaded %zu class policies into CollisionEnvSemantic.", policies_.size());
}

void CollisionEnvSemantic::checkSelfCollision(
  const collision_detection::CollisionRequest & /*req*/,
  collision_detection::CollisionResult & res,
  const moveit::core::RobotState & /*state*/) const
{
  // TODO(P1.4): delegate to FCL self-collision; for now mark no collision.
  res.collision = false;
  res.distance = std::numeric_limits<double>::infinity();
}

void CollisionEnvSemantic::checkSelfCollision(
  const collision_detection::CollisionRequest & req,
  collision_detection::CollisionResult & res,
  const moveit::core::RobotState & state,
  const collision_detection::AllowedCollisionMatrix & /*acm*/) const
{
  checkSelfCollision(req, res, state);
}

void CollisionEnvSemantic::checkRobotCollision(
  const collision_detection::CollisionRequest & /*req*/,
  collision_detection::CollisionResult & res,
  const moveit::core::RobotState & /*state*/) const
{
  // TODO(P1.4): for each link's bounding shape, query per-class ESDF.
  // Apply policy:
  //   hard      -> if distance < padding, mark res.collision = true
  //   soft      -> accumulate cost into res.distance (lower = closer to coll.)
  //   attractor -> negative cost contribution (pull planner inward)
  res.collision = false;
  res.distance = std::numeric_limits<double>::infinity();
}

void CollisionEnvSemantic::checkRobotCollision(
  const collision_detection::CollisionRequest & req,
  collision_detection::CollisionResult & res,
  const moveit::core::RobotState & state,
  const collision_detection::AllowedCollisionMatrix & /*acm*/) const
{
  checkRobotCollision(req, res, state);
}

void CollisionEnvSemantic::checkRobotCollision(
  const collision_detection::CollisionRequest & req,
  collision_detection::CollisionResult & res,
  const moveit::core::RobotState & /*state1*/,
  const moveit::core::RobotState & state2) const
{
  // TODO(P1.4): swept-volume query through per-class ESDFs.
  // Skeleton: delegate to the single-state check at state2.
  checkRobotCollision(req, res, state2);
}

void CollisionEnvSemantic::checkRobotCollision(
  const collision_detection::CollisionRequest & req,
  collision_detection::CollisionResult & res,
  const moveit::core::RobotState & /*state1*/,
  const moveit::core::RobotState & state2,
  const collision_detection::AllowedCollisionMatrix & /*acm*/) const
{
  checkRobotCollision(req, res, state2);
}

void CollisionEnvSemantic::distanceSelf(
  const collision_detection::DistanceRequest & /*req*/,
  collision_detection::DistanceResult & res,
  const moveit::core::RobotState & /*state*/) const
{
  res.minimum_distance.distance = std::numeric_limits<double>::infinity();
}

void CollisionEnvSemantic::distanceRobot(
  const collision_detection::DistanceRequest & /*req*/,
  collision_detection::DistanceResult & res,
  const moveit::core::RobotState & /*state*/) const
{
  // TODO(P1.4): return min over hard classes; ignore soft/attractor here.
  res.minimum_distance.distance = std::numeric_limits<double>::infinity();
}

void CollisionEnvSemantic::setWorld(const collision_detection::WorldPtr & world)
{
  collision_detection::CollisionEnv::setWorld(world);
}

}  // namespace scout_piper_scene_repr

// pluginlib export
PLUGINLIB_EXPORT_CLASS(
  scout_piper_scene_repr::SemanticCollisionDetectorAllocator,
  collision_detection::CollisionDetectorAllocator)
