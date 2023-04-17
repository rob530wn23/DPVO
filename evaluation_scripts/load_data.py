import cv2
import glob
import os
import datetime
import numpy as np
import os.path as osp
from pathlib import Path

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

# import evo
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.trajectory import PoseTrajectory3D
# from evo.tools import plot
from evo.core.metrics import PoseRelation

test_split = \
    ["MH%03d"%i for i in range(8)] + \
    ["ME%03d"%i for i in range(8)]

STRIDE = 1
fx, fy, cx, cy = [320, 320, 320, 240]

def make_traj(args) -> PoseTrajectory3D:
    if isinstance(args, tuple):
        traj, tstamps = args
        return PoseTrajectory3D(positions_xyz=traj[:,:3], orientations_quat_wxyz=traj[:,3:], timestamps=tstamps)
    # print("type(args): ", type(args))
    # print("args: ", args)
    # print("isinstance(args, PoseTrajectory3D): ", isinstance(args, PoseTrajectory3D))
    assert isinstance(args, PoseTrajectory3D), type(args)
    return deepcopy(args)

def ate(traj_ref, traj_est, timestamps):

    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=timestamps)

    traj_ref = PoseTrajectory3D(
        positions_xyz=traj_ref[:,:3],
        orientations_quat_wxyz=traj_ref[:,3:],
        timestamps=timestamps)
    
    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True, align_origin=True)

    return result.stats["rmse"]

if __name__ == '__main__':

    results = {}
    all_results = []

    # load ground truth (reference) trajectory
    traj_ref = 'P006/pose_left.txt'
    traj_ref = np.loadtxt(traj_ref, delimiter=" ")

    # load DPVO (estimate) trajectory 
    traj_est= np.loadtxt('ransac_results/DPVO/poses9.txt', dtype=np.float64)
    tstamps = np.loadtxt('ransac_results/DPVO/tstamps9.txt', dtype=np.float64)

    # do evaluation
    ate_score = ate(traj_ref, traj_est, tstamps)
    print(ate_score)

    pred_traj = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=tstamps)
    
    gt_traj = PoseTrajectory3D(
        positions_xyz=traj_ref[:,:3],
        orientations_quat_wxyz=traj_ref[:,3:],
        timestamps=tstamps)

    pred_traj = make_traj(pred_traj)
    gt_traj = make_traj(gt_traj)

    gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

    pred_traj.align(gt_traj, correct_scale=True)

    x_gt = np.array(gt_traj.positions_xyz[:,0])
    y_gt = np.array(gt_traj.positions_xyz[:,1])
    x_pred = np.array(pred_traj.positions_xyz[:,0])
    y_pred = np.array(pred_traj.positions_xyz[:,1])
    
    # np.savetxt('x_gt.txt', x_gt)
    # np.savetxt('y_gt.txt', y_gt)
    np.savetxt('x_pred_of10.txt', x_pred)
    np.savetxt('y_pred_of10.txt', y_pred)


    # print(traj_est.positions_xyz)
    # plt.plot(gt_traj.positions_xyz[:,0], gt_traj.positions_xyz[:,1], label = "Ground Truth")
    # plt.plot(np.array([1, 2]), np.array([1, 2]), label = "Patch Improvement")
    # plt.legend()
    # plt.show()