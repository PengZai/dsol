import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# EuRoC MAV Dataset calib.txt: 435.2046959714599 435.2046959714599 367.4517211914062 252.2008514404297 0.11007784219

import rosbag
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge


if __name__ == '__main__':
    bag_file = "/root/datasets2/polytunnel_2025_May/Easy.bag"
    bag_name = "Easy_rectified"
    bag = rosbag.Bag(bag_file, "r")
    bridge = CvBridge()
    count = 0
    topics = [
        "/zed2i/zed_node/left_raw/image_raw_color/compressed",
        "/zed2i/zed_node/right_raw/image_raw_color/compressed"
    ]

    data_dir = Path("/root/datasets2/polytunnel_2025_May") / bag_name
    left_dir = data_dir / "left_raw_rectified"
    right_dir = data_dir / "right_raw_rectified"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    
    # dalsa_rgb0
    K1 = np.array([[1071.58, 0.0, 955.33],[0.0, 1072.15, 555.983], [0.0, 0.0, 1.0]])
    D1 = np.array([[-0.0519091, 0.0250782, 0.000533807, 0.000117517, -0.0101723]])

    # dalsa_rgb1
    K2 = np.array([[1067.61, 0.0, 964.59],[0.0, 1068.32, 555.556], [0.0, 0.0, 1.0]])
    D2 = np.array([[-0.0500384, 0.0221768, 0.000412861, 0.000442194, -0.00894885]])

    T12 = np.array([[0.99996212, -0.00133436,  0.00860078, -0.119729],
                  [0.00134459,  0.99999839, -0.00118413, -0.000135398],
                  [-0.00859918,  0.00119565,  0.99996231, 0.000681149],
                  [0.0,0.0,0.0,1.0]
                  ])
    T21 = np.linalg.inv(T12.copy())
    R =  T21[:3, :3]
    t =  T21[:3, 3]
    image_size = (1920, 1080)
    print(f" T21:\n{T21}")

    # Stereo rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, t,
        flags=cv2.CALIB_ZERO_DISPARITY,   # Make principal points aligned
        alpha=0,                          # Cropping: 0 = zoom in to valid pixels only
        newImageSize=image_size
    )

    print(f"new rotation matrix R1:\n{R1}")
    print(f"new intrinsics K1:\n{P1}")
    print(f"new rotation matrix R2:\n{R2}")
    print(f"new intrinsics K2:\n{P2}")

    leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)


    for topic, msg, t in tqdm(bag.read_messages(topics=topics)):
        count = t.to_nsec() // 1000000
        timestamp_ns = t.to_nsec()


        # cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)  # or cv2.IMREAD_COLOR


        if topic == topics[0]:
            left_rectified = cv2.remap(cv_img, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            cv2.imwrite(str(left_dir / f"{timestamp_ns}.png"), left_rectified)
        elif topic == topics[1]:
            right_rectified = cv2.remap(cv_img, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            cv2.imwrite(str(right_dir / f"{timestamp_ns}.png"), right_rectified)
    bag.close()
