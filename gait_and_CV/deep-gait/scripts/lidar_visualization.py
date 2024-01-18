import open3d as o3d
import pyrealsense2 as rs
import numpy as np

def get_lidar_data():
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    pipeline.start(config)

    try:
                while True:
                    # Wait for the next set of frames from the camera
                    frames = pipeline.wait_for_frames()

                    # Get depth frame
                    depth_frame = frames.get_depth_frame()
                    if not depth_frame:
                        continue

                    # Convert RealSense depth frame to Numpy array
                    depth_image = np.asanyarray(depth_frame.get_data())

                    # Return the depth image
                    return depth_image

    finally:
        pipeline.stop()

def visualize_point_cloud(depth_image):
    # Create point cloud from depth image
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.Image(depth_image),
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    )

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # Get LiDAR data
    lidar_data = get_lidar_data()

    # Visualize point cloud
    visualize_point_cloud(lidar_data)
