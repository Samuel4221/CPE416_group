import rosbag2_py
import numpy as np
import matplotlib.pyplot as plt

# open bag
reader = rosbag2_py.SequentialReader()
reader.open(...)

# read messages and store values
xs, ys, thetas, t = [], [], [], []

while reader.has_next():
    topic, data, stamp = reader.read_next()
    if topic == '/ekf_odom':
        odom = deserialize_odometry(data)
        xs.append(odom.pose.pose.position.x)
        ys.append(odom.pose.pose.position.y)
        thetas.append(yaw_from_quat(odom.pose.pose.orientation))
        t.append(stamp)

# plot
plt.plot(xs, ys)
plt.show()
