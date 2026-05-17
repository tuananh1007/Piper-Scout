"""Segmentation node — ROS 2 Humble port skeleton.

Ports stem_grasp_ros1/scripts/segmentation_node.py to rclpy.
Publishes:
    /stem_grasp/stem_mask    (sensor_msgs/Image)
    /stem_grasp/leaf_mask    (sensor_msgs/Image)
    /stem_grasp/target_mask  (sensor_msgs/Image) — when target_caption set
    /stem_grasp/target_point (geometry_msgs/PointStamped) — 3D centroid of target
Subscribes:
    /camera/color/image_raw  (sensor_msgs/Image)
    /camera/depth/image_rect_raw  (sensor_msgs/Image)
    /joint_states  (sensor_msgs/JointState) — for stationary gate

TODO(port-from-ros1): wire up the YOLO + Grounded-SAM stack from the ROS 1 file.
"""

import rclpy
from rclpy.node import Node


class SegmentationNode(Node):
    def __init__(self) -> None:
        super().__init__("stem_grasp_segmentation")
        self.get_logger().info(
            "segmentation_node (ROS 2 port skeleton) ready — algorithm TODO."
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
