#pragma once

// Semantic collision detector for MoveIt 2 (P1.3.1 / P1.7.7).
//
// CollisionEnvSemantic is MoveIt's FCL environment (self-collision and
// planning-scene objects unchanged) plus a check of the robot against the
// semantic distance field published by scene_query_node:
//
//   * every link's collision shapes are covered by spheres once per robot
//     model (bounding cylinder -> overlapping spheres, see
//     semantic_distance_field.hpp);
//   * each sphere (radius + link padding + sphere_padding_m) is checked with
//     SemanticDistanceField::checkSphere: hard classes with their padding, the
//     leaf penetration cap, and unknown / stale voxels as obstacles;
//   * outside the field's grid the field says nothing and only FCL applies;
//   * with require_field (default) a missing, old or untransformable field
//     makes every robot check report a collision, so planning stops instead
//     of ignoring the plant.
//
// Select it in move_group with `collision_detector: "Semantic"` (pluginlib class
// SemanticCollisionPluginLoader, see plugin_description.xml).
//
// Robot-vs-field pairs use the body name "semantic". Allow ("<link>",
// "semantic") in the AllowedCollisionMatrix, or list the link in ignore_links,
// to exempt a link (for example the fingers during the final grasp).

#include <memory>
#include <string>
#include <vector>

#include <moveit/collision_detection/collision_detector_allocator.h>
#include <moveit/collision_detection/collision_plugin.h>
#include <moveit/collision_detection_fcl/collision_env_fcl.h>

#include "scout_piper_scene_repr/semantic_distance_field.hpp"

namespace scout_piper_scene_repr
{

/// Body name used for the field in contacts, distance results and the ACM.
inline const std::string kSemanticBody = "semantic";

/// Spheres covering one collision shape, in that shape's frame.
struct LinkShapeSpheres
{
  const moveit::core::LinkModel * link{nullptr};
  std::size_t shape_index{0};
  std::vector<Sphere> spheres;
};

/// Sphere cover of every link with collision geometry (no padding applied).
std::vector<LinkShapeSpheres> computeRobotSpheres(const moveit::core::RobotModelConstPtr & model);

class CollisionEnvSemantic : public collision_detection::CollisionEnvFCL
{
public:
  explicit CollisionEnvSemantic(
    const moveit::core::RobotModelConstPtr & robot_model, double padding = 0.0,
    double scale = 1.0);

  CollisionEnvSemantic(
    const moveit::core::RobotModelConstPtr & robot_model,
    const collision_detection::WorldPtr & world, double padding = 0.0, double scale = 1.0);

  CollisionEnvSemantic(
    const CollisionEnvSemantic & other, const collision_detection::WorldPtr & world);

  ~CollisionEnvSemantic() override = default;

  using collision_detection::CollisionEnvFCL::distanceRobot;

  void checkRobotCollision(
    const collision_detection::CollisionRequest & req,
    collision_detection::CollisionResult & res,
    const moveit::core::RobotState & state) const override;

  void checkRobotCollision(
    const collision_detection::CollisionRequest & req,
    collision_detection::CollisionResult & res,
    const moveit::core::RobotState & state,
    const collision_detection::AllowedCollisionMatrix & acm) const override;

  /// FCL has no continuous check on Humble; both are sampled along the segment
  /// every ``continuous_step`` of RobotState::distance.
  void checkRobotCollision(
    const collision_detection::CollisionRequest & req,
    collision_detection::CollisionResult & res,
    const moveit::core::RobotState & state1,
    const moveit::core::RobotState & state2) const override;

  void checkRobotCollision(
    const collision_detection::CollisionRequest & req,
    collision_detection::CollisionResult & res,
    const moveit::core::RobotState & state1,
    const moveit::core::RobotState & state2,
    const collision_detection::AllowedCollisionMatrix & acm) const override;

  void distanceRobot(
    const collision_detection::DistanceRequest & req,
    collision_detection::DistanceResult & res,
    const moveit::core::RobotState & state) const override;

private:
  std::shared_ptr<const std::vector<LinkShapeSpheres>> spheres_;

  void checkDiscrete(
    const collision_detection::CollisionRequest & req,
    collision_detection::CollisionResult & res,
    const moveit::core::RobotState & state,
    const collision_detection::AllowedCollisionMatrix * acm) const;

  void checkContinuous(
    const collision_detection::CollisionRequest & req,
    collision_detection::CollisionResult & res,
    const moveit::core::RobotState & state1,
    const moveit::core::RobotState & state2,
    const collision_detection::AllowedCollisionMatrix * acm) const;

  void checkField(
    const collision_detection::CollisionRequest & req,
    collision_detection::CollisionResult & res,
    const moveit::core::RobotState & state,
    const collision_detection::AllowedCollisionMatrix * acm) const;

  bool exempt(
    const std::string & link,
    const collision_detection::AllowedCollisionMatrix * acm) const;
};

/// Allocator for CollisionEnvSemantic (name "Semantic").
class SemanticCollisionDetectorAllocator
  : public collision_detection::CollisionDetectorAllocatorTemplate<
    CollisionEnvSemantic, SemanticCollisionDetectorAllocator>
{
public:
  static const std::string NAME;
};

/// The pluginlib class MoveIt's CollisionPluginLoader instantiates for
/// `collision_detector: "Semantic"`; it installs the allocator on the scene.
class SemanticCollisionPluginLoader : public collision_detection::CollisionPlugin
{
public:
  bool initialize(const planning_scene::PlanningScenePtr & scene) const override;
};

}  // namespace scout_piper_scene_repr
