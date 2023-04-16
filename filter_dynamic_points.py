import argparse
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch

# Tartan params
fx = 320.0  # focal length x
fy = 320.0  # focal length y
cx = 320.0  # optical center x
cy = 240.0  # optical center y
cam_intrinsic_mat = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

fov = 90 # field of view

width = 640
height = 480

def skew_symmetric(v):
    """Compute the skew-symmetric matrix of a 3x1 vector."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def essential_matrix(R, t):
    """Compute the essential matrix from a rotation matrix and a translation vector."""
    tx = skew_symmetric(t)
    E = tx @ R
    return E

def get_pixel_coord_in_img2(E, x1, y1, row, col):
    # Compute the epipolar line in image 2
    pt1 = np.array([x1, y1, 1])
    line2 = np.dot(E, pt1)

    # Find the intersection of the epipolar line and the image plane of camera 2
    x2_1 = 0
    x2_2 = col - 1
    y2_1 = int(-(line2[2] + line2[0]*x2_1) / line2[1])
    y2_2 = int(-(line2[2] + line2[0]*x2_2) / line2[1])

    # The intersection point corresponds to the pixel coordinate of P in image 2
    x2 = (x2_1 + x2_2) // 2
    y2 = (y2_1 + y2_2) // 2
    
    return x2, y2

def compute_epipolar_line(p1, E, K):
    """
    Compute the epipolar line in image 2 for a given pixel coordinate in image 1,
    essential matrix E, and camera intrinsic matrix K.

    :param p1: Pixel coordinate in image 1 (numpy array of shape (2,))
    :param E: Essential matrix (numpy array of shape (3, 3))
    :param K: Camera intrinsic matrix (numpy array of shape (3, 3))
    :return: Epipolar line coefficients (numpy array of shape (3,)) (under normalized coordinate ? )
    """
    # Convert pixel coordinate to homogeneous coordinate
    p1_h = np.array([p1[0], p1[1], 1.0])

    # Convert pixel coordinate in image 1 to normalized image coordinate
    p1_normalized = np.linalg.inv(K) @ p1_h

    # Compute the epipolar line in image 2
    l2 = E @ p1_normalized

    return l2


# plot optical flow (downsample)
def plot_optical_flow_all_pts(img, flow, window_name, flow_dist_thresh=2, window_size=(640, 480)):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_size)

    # Display the optical flow
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h:10, 0:w:10].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    # vis = cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
    vis = img

    # Highlight the points on image 1 and image 2 with red and blue, respectively
    for i in range(len(x)):
        point1 = (x[i], y[i])
        point2 = (x[i] + int(fx[i]), y[i] + int(fy[i]))
        cv2.circle(vis, point1, 3, (0, 0, 255), -1)
        cv2.circle(vis, point2, 3, (255, 0, 0), -1)

    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    cv2.imshow(window_name, vis)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

# plot optical flow, only plot points with larger move than thresh, and donw sample points
def plot_optical_flow(img, flow, window_name, flow_dist_thresh=2, visualizatino_scale=1, window_size=(640, 480)):
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(window_name, window_size[0]*visualizatino_scale, window_size[1]*visualizatino_scale)

    # Display the optical flow
    h, w = img.shape[:2]
    # y, x = np.mgrid[0:h, 0:w].reshape(2, -1).astype(int)
    # downsample points
    y, x = np.mgrid[0:h:20, 0:w:20].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    dist = np.sqrt(fx ** 2 + fy ** 2)
    mask = dist > flow_dist_thresh
    lines = np.vstack([x[mask], y[mask], x[mask] + fx[mask], y[mask] + fy[mask]]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    # vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis = img.copy()

    # Highlight the points on image 1 and image 2 with red and blue, respectively
    for i in range(len(x)):
        if dist[i] > flow_dist_thresh:
            point1 = (x[i], y[i])
            point2 = (x[i] + int(fx[i]), y[i] + int(fy[i]))
            cv2.circle(vis, point1, 3, (0, 255, 0), -1) 
            cv2.circle(vis, point2, 3, (255, 0, 0), -1)

    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    # cv2.imshow(window_name, vis)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return vis


# compute optical flow
def compute_optical_flow(img1_path, img2_path):
    # Load two images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calculate the optical flow using the Lucas-Kanade method
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Threshold the flow vectors based on their magnitude
    flow_magnitude = np.sqrt(np.sum(flow**2, axis=2))
    valid_flow = flow_magnitude > 1.0  # set the threshold to 1.0
    return flow, valid_flow

def point_line_distance(point, line_coeffs):
    """
    Computes the distance from a 2D point to a 2D line, given the line coefficients.
    """
    # Unpack the line coefficients
    a, b, c = line_coeffs

    # Compute the distance from the point to the line
    distance = abs(a * point[0] + b * point[1] + c) / np.sqrt(a ** 2 + b ** 2)

    return distance

# =========================== verify optical flow
def verify_tartan_optical_flow(pre_id, cur_id, data_dir):
    # pre_id = "000001"
    # cur_id = "000002"
    # Tartan optical flow
    flow_gt = np.load(data_dir + "/flow/{}_{}_flow.npy".format(pre_id, cur_id))
    # for row in flow:
        # print(row)
    print("flow gt shape", flow_gt.shape)
    img1_path = data_dir + "/image_left/{}_left.png".format(pre_id)
    img2_path = data_dir + "/image_left/{}_left.png".format(cur_id)
    cal_flow, _ = compute_optical_flow(img1_path, img2_path)
    
    img1 = cv2.imread(img1_path)
    cal_flow_img = plot_optical_flow(img1, cal_flow, 'cal flow')
    gt_flow_img = plot_optical_flow(img1, flow_gt, 'gt flow')
    concatenated_image = cv2.hconcat([cal_flow_img, gt_flow_img])

    cv2.imwrite("./flow_test/{}_flow.png".format(pre_id), concatenated_image)
    # cv2.imshow("flow", concatenated_image)

    # cv2.waitKey()
    # cv2.destroyAllWindows()


def load_poses(pose_file):
    poses = np.loadtxt(pose_file)
    # print(poses)
    Rs = []
    ts = []
    for pose in poses:
        ts.append(pose[0:3])
        # qs.append(pose[3:])
        R = Rotation.from_quat(np.array([pose[3], pose[4], pose[5], pose[6]])).as_matrix()
        Rs.append(R)
    return Rs, ts

def T_inv(T):
    # Compute the inverse of the rotation matrix (transpose of the top-left 3x3 submatrix)
    R_inv = T[:3, :3].T

    # Compute the inverse translation vector (-R_inv * t)
    t_inv = -R_inv @ T[:3, 3]

    # Create a 4x4 identity matrix
    T_inv = np.identity(4)

    # Set the top-left 3x3 submatrix to the inverse rotation matrix R_inv
    T_inv[:3, :3] = R_inv

    # Set the top-right 3x1 submatrix to the inverse translation vector t_inv
    T_inv[:3, 3] = t_inv

    # print(T)
    # print(T_inv)
    return T_inv

def T(R, t):
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def pixel_to_normalized_coord(u, v, fx, fy, cx, cy):
    # Convert pixel coordinates to normalized image coordinates
    x = (u - cx) / fx
    y = (v - cy) / fy
    return np.array([x, y])

def load_flow(frame_id_1, frame_id_2, data_dir):
    return load_flow_from_file(data_dir + "flow/{}_{}_flow.npy".format(frame_id_1, frame_id_2))

def load_flow_from_file(flow_file):
    flow_gt = np.load(flow_file)
    return flow_gt

def number_to_frame_id(num:int) -> str:
    s = "{:06d}".format(num)
    return s

import matplotlib.pyplot as plt

def plot_distribution(data, bins=50):
    # Create histogram plot of the data
    plt.hist(data, bins=bins)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Values')
    plt.show()

def load_depth_map(dir):
    depth_map = np.load(dir)
    return depth_map

def main():
    # the key threshold value: 5 is the best experimental value
    epipolar_dist_thresh = 5.5 / fx   # TODO
    print("epipolar_dist_thresh is ", epipolar_dist_thresh)

    parser = argparse.ArgumentParser(description='Filter optical flow for Tartan.')
    parser.add_argument('data_dir', type=str, help='Path to the data folder.')

    args = parser.parse_args()

    data_dir = args.data_dir
    Rs, ts = load_poses(data_dir + "pose_left.txt")
    # print("Rs len", len(Rs), "ts len", len(ts))

    for i in range(1, len(Rs)):
        cur_id = number_to_frame_id(i)
        pre_id = number_to_frame_id(i - 1)

        # verify_tartan_optical_flow(pre_id, cur_id, data_dir)
        # continue

        R_cur = Rs[i]
        t_cur = ts[i]
        T_cur = T(R_cur, t_cur)

        R_pre = Rs[i - 1]
        t_pre = ts[i - 1]
        T_pre = T(R_pre, t_pre)

        T_pre_cur = T_inv(T_pre) @ T_cur
        T_cur_pre = T_inv(T_cur) @ T_pre

        E = essential_matrix(T_pre_cur[:3, :3], T_pre_cur[:3, 3])
        print("---- frame cur {} pre {} ".format(cur_id, pre_id))
        
        # gt flow
        flow = load_flow(pre_id, cur_id, data_dir)
        # print("flow shape:", flow.shape)

        # calculated flow
        img1_path = data_dir + "image_left/{}_left.png".format(pre_id)
        img2_path = data_dir + "image_left/{}_left.png".format(cur_id)

        depth_pre = load_depth_map(data_dir + "depth_left/{}_left_depth.npy".format(pre_id))
        depth_cur = load_depth_map(data_dir + "depth_left/{}_left_depth.npy".format(cur_id))
        # print("depth_pre shape:", depth_pre.shape)
        # print("depth_cur shape:", depth_cur.shape)

        depth_diff_thresh = 2.0  # TODO

        # visualization before filtering dynamic objects
        img1 = cv2.imread(img1_path)
        flow_img_before = plot_optical_flow(img1, flow, "flow {}_{}".format(pre_id, cur_id))

        # filter invalid flows with big epipolar distance
        # Note: this method to filter dynamic objects is based on epipolar geometry, it will degenerate under pure rotation or large rotation
        epipolar_dists = []
        filtered_flow_coordinates = []
        count_valid = 0
        count_depth_selected = 0
        count_epipolar_selected = 0
        count_not_valid = 0
        x = np.zeros((flow.shape[0], flow.shape[1]))
        y = np.zeros((flow.shape[0], flow.shape[1]))
        for row in range(0, flow.shape[0]):
            for col in range(0, flow.shape[1]):
                x[row, col] = flow[row, col][0]
                y[row, col] = flow[row, col][1]
        x_avg = np.mean(x)
        y_avg = np.mean(y)
        mangitude_avg = np.sqrt(x_avg ** 2 + y_avg ** 2)

        for row in range(0, flow.shape[0]):
            for col in range(0, flow.shape[1]):
                dx, dy = flow[row, col]
                # whether is this point flow valid (invalid if too small)
                dist = np.sqrt( (dx-x_avg) ** 2 + (dy-y_avg) ** 2)
                # print(dx, x_avg, dy, y_avg, dist, mangitude_avg)
                if dist < 1.5 * mangitude_avg:
                    count_not_valid += 1
                    continue
                else:
                    count_valid += 1

                # Check the difference in depth values between the corresponding points
                depth_pre_value = depth_pre[row, col]
                row_new = np.min((int(row + dy), 479))
                col_new = np.min((int(col + dx), 639))
                depth_cur_value = depth_cur[row_new, col_new]
                depth_diff = abs(depth_pre_value - depth_cur_value)

                # print("depth_diff", depth_diff)
                if depth_diff > depth_diff_thresh:
                    # Large depth difference, consider as dynamic object
                    count_depth_selected += 1
                    filtered_flow_coordinates.append(np.array([row, col]))
                    flow[row, col][0] = 0
                    flow[row, col][1] = 0
                else:
                    p_pre = np.array([col, row])
                    p_cur = np.array([col + dx, row + dy])
                    epipolar_line = compute_epipolar_line(p_pre, E, cam_intrinsic_mat)
                    p_cur_normalized = pixel_to_normalized_coord(p_cur[0], p_cur[1], fx, fy, cx, cy)
                    d = point_line_distance(p_cur_normalized, epipolar_line)
                    epipolar_dists.append(d)
                    #print(d, epipolar_dist_thresh)
                    if d > epipolar_dist_thresh:
                        count_epipolar_selected += 1
                        filtered_flow_coordinates.append(np.array([row, col]))
                        flow[row, col][0] = 0
                        flow[row, col][1] = 0
        print("count_valid: ", count_valid)
        print("count_not_valid: ", count_not_valid)
        print("count_depth_selected: ", count_depth_selected)
        print("count_epipolar_selected: ", count_epipolar_selected)
        print("x = ", np.shape(flow))
        # print(epipolar_dists)
        # visualization after filtering dynamic objects
        filter_mask = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.uint8)
        vis = img1
        for coor in filtered_flow_coordinates:
            # cv2.circle(vis, (coor[1], coor[0]), 3, (0, 0, 255), -1)
            filter_mask[coor[0], coor[1]] = 255
        
        np.save('mask_index/filtered_flow_coordinates{}.npy'.format(i), filtered_flow_coordinates)
        #change blue channel 
        vis[filter_mask == 255, 0] = 255


        concatenated_image = cv2.hconcat([flow_img_before, vis])
        cv2.imwrite("./result/{}_result.png".format(cur_id), concatenated_image)



if __name__ == '__main__':
    '''
    Usage: python3 filter_dynamic_points.py P006/
    '''
    main()

