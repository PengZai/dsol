import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# EuRoC MAV Dataset calib.txt: 435.2046959714599 435.2046959714599 367.4517211914062 252.2008514404297 0.11007784219

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

if __name__ == '__main__':
    bag_file = "/root/datasets2/BotanicGarden/1005-07/1005_07_img10hz600p.bag"
    bag_name = "1018_00_img10hz600p_rectified"
    bag = rosbag.Bag(bag_file, "r")
    bridge = CvBridge()
    count = 0
    topics = [
        "/dalsa_rgb/left/image_raw",
        "/dalsa_rgb/right/image_raw"
    ]

    data_dir = Path("/root/datasets2/BotanicGarden/1005-07/1005_07_img10hz600p") / bag_name
    left_dir = data_dir / "left_rgb_rectified"
    right_dir = data_dir / "right_rgb_rectified"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    
    # dalsa_rgb0
    K1 = np.array([[642.9165664800531, 0.0, 460.1840658156501],[0.0, 641.9171825800378, 308.5846449100310], [0.0, 0.0, 1.0]])
    D1 = np.array([[-0.060164620903866, 0.094005180631043, 0, 0, 0.0]])

    # dalsa_rgb1
    K2 = np.array([[644.4385505412966, 0.0, 455.1775919513420],[0.0, 643.5879520187435, 304.1616226347153], [0.0, 0.0, 1.0]])
    D2 = np.array([[-0.057705696896734, 0.086955444511364, 0, 0, 0.0]])

    T12 = np.array([[0.999994564612669,-0.00327143011166783,-0.000410475508767800,0.253736175410149],
                  [0.00326819763481066,0.999965451959397,-0.00764289028177120,-0.000362553856124796],
                  [0.000435464509051199,0.00764150722461529,0.999970708440001,-0.000621002717451192],
                  [0.0,0.0,0.0,1.0]
                  ])
    T21 = np.linalg.inv(T12.copy())
    R =  T21[:3, :3]
    t =  T21[:3, 3]
    image_size = (960, 600)
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


        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if topic == topics[0]:
            left_rectified = cv2.remap(cv_img, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            cv2.imwrite(str(left_dir / f"{timestamp_ns}.png"), left_rectified)
        elif topic == topics[1]:
            right_rectified = cv2.remap(cv_img, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            cv2.imwrite(str(right_dir / f"{timestamp_ns}.png"), right_rectified)
    bag.close()
