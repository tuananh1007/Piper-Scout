"""Point cloud filter node — ROS 2 Humble port skeleton.

Ports stem_grasp_ros1/scripts/pointcloud_node.py.
Filters the RealSense cloud by per-class masks (stem/leaf/target), and
publishes /static_cloud_out for the MoveIt collision world (Phase 0).
Phase 1 replaces this with the semantic nvblox pipeline.
"""

import rclpy
from rclpy.node import Node


class PointCloudNode(Node):
    def __init__(self) -> None:
        super().__init__("stem_grasp_pointcloud")
        self.get_logger().info(
            "pointcloud_node (ROS 2 port skeleton) ready — algorithm TODO."
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PointCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
