#pragma once

// Semantic collision detector for MoveIt 2.
//
// Backed by N parallel nvblox ESDFs (one per semantic class) consumed via the
// SDF-query API. Class behaviors (hard / soft / attractor) are loaded from
// semantic_classes.yaml at construction time.
//
// This header declares the public allocator + env classes; the implementation
// lives in semantic_collision_plugin.cpp.

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <moveit/collision_detection/collision_env.h>
#include <moveit/collision_detection/collision_detector_allocator.h>

namespace scout_piper_scene_repr
{

enum class ClassBehavior
{
  Hard,
  Soft,
  Attractor,
};

struct ClassPolicy
{
  ClassBehavior behavior{ClassBehavior::Hard};
  double padding_m{0.0};

  // Used when behavior == Soft
  double cost_weight{0.0};
  double max_penetration_m{0.02};

  // Used when behavior == Attractor
  double attract_radius_m{0.10};
  double attract_weight{-20.0};
};

using ClassPolicyMap = std::unordered_map<std::string, ClassPolicy>;

/**
 * @brief MoveIt 2 collision environment that consults per-class nvblox ESDFs.
 *
 * Phase 1 skeleton — most methods delegate to a sibling FCL env for self-collision
 * while their robot-world implementations are still TODO. Filling these in is
 * tracked under P1.4 in PROGRESS.md.
 */
class CollisionEnvSemantic : public collision_detection::CollisionEnv
{
public:
  // The MoveIt allocator template needs single-arg and two-arg constructors
  // that take only the robot model (and optionally the world). Policies are
  // loaded post-construction via setPolicies() so they don't need to be
  // baked into the constructor signature.
  explicit CollisionEnvSemantic(
    const moveit::core::RobotModelConstPtr & robot_model);

  CollisionEnvSemantic(
    const moveit::core::RobotModelConstPtr & robot_model,
    const collision_detection::WorldPtr & world);

  ~CollisionEnvSemantic() override = default;

  /// Load per-class policies (typically from semantic_classes.yaml).
  /// Called by the wrapper service that bridges /scene_repr/policy ->
  /// the active CollisionEnv instance(s).
  void setPolicies(const ClassPolicyMap & policies);

  void checkSelfCollision(
    const collision_detection::CollisionRequest & req,
    collision_detection::CollisionResult & res,
    const moveit::core::RobotState & state) const override;

  void checkSelfCollision(
    const collision_detection::CollisionRequest & req,
    collision_detection::CollisionResult & res,
    const moveit::core::RobotState & state,
    const collision_detection::AllowedCollisionMatrix & acm) const override;

  void checkRobotCollision(
    const collision_detection::CollisionRequest & req,
    collision_detection::CollisionResult & res,
    const moveit::core::RobotState & state) const override;

  void checkRobotCollision(
    const collision_detection::CollisionRequest & req,
    collision_detection::CollisionResult & res,
    const moveit::core::RobotState & state,
    const collision_detection::AllowedCollisionMatrix & acm) const override;

  // Continuous-collision overloads — required pure virtuals on MoveIt Humble.
  // P1.4 will fill these in with swept-volume queries against the per-class
  // ESDFs; for now they conservatively delegate to the single-state version
  // at state2 (the planner's "next" state) so trajectories see at least the
  // same collisions as a per-waypoint check would.
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

  void distanceSelf(
    const collision_detection::DistanceRequest & req,
    collision_detection::DistanceResult & res,
    const moveit::core::RobotState & state) const override;

  void distanceRobot(
    const collision_detection::DistanceRequest & req,
    collision_detection::DistanceResult & res,
    const moveit::core::RobotState & state) const override;

  void setWorld(const collision_detection::WorldPtr & world) override;

private:
  ClassPolicyMap policies_;

  // TODO(P1.4): hold per-class ESDF clients here. For Phase 1 v0 these will be
  // rclcpp subscriptions to /scene_repr/esdf/<class>; for v1 they become
  // synchronous service queries against the forked nvblox.
  // std::vector<EsdfClient> esdf_clients_;
};

/**
 * @brief Allocator exported via pluginlib for the MoveIt collision_detector.
 */
class SemanticCollisionDetectorAllocator
  : public collision_detection::CollisionDetectorAllocatorTemplate<
      CollisionEnvSemantic,
      SemanticCollisionDetectorAllocator>
{
public:
  static const std::string NAME;
};

}  // namespace scout_piper_scene_repr
