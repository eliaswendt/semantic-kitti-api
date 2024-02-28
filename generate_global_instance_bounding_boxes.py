import argparse
import os
import shutil
import time
import numpy as np
np.seterr(all='raise') # required for catching np warnings

from collections import defaultdict
from numpy.linalg import inv
from tqdm import tqdm

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from sklearn.cluster import DBSCAN

def parse_calibration(filename):
  """ read calibration file with given filename

      Returns
      -------
      dict
          Calibration matrices as 4x4 numpy arrays.
  """
  calib = {}

  calib_file = open(filename)
  for line in calib_file:
    key, content = line.strip().split(":")
    values = [float(v) for v in content.strip().split()]

    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0

    calib[key] = pose

  calib_file.close()

  return calib


# cf https://github.com/PRBonn/semantic-kitti-api/issues/78
def parse_poses(filename, calibration):
  """ read poses file with per-scan poses from given filename

      Returns
      -------
      list
          list of poses as 4x4 numpy arrays.
  """
  file = open(filename)

  poses = []

  Tr = calibration["Tr"]
  Tr_inv = inv(Tr)

  for line in file:
    values = [float(v) for v in line.strip().split()]

    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0

    poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr))) 

  return poses


def translate_points_to_pose(base_pose_inv, current_pose, points_original):
  # transform from current_pose to base_pose
  current_to_base = np.matmul(base_pose_inv, current_pose)

  points_translated = np.ones((points_original.shape[0], 4), dtype=np.float32)
  points_translated[:, 0:3] = points_original[:, 0:3]
  remissions = points_original[:, 3]

  # apply pose transformation to points
  points_translated = np.matmul(current_to_base, points_translated.T).T
  points_translated[:, 3] = remissions # re-add remissions

  return points_translated.astype(np.float32)


def plot_points_3d(points):
  # in case scan.shape is (,4)
  points = points.reshape((-1, 4))

  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  ax.scatter(points[:,0], points[:,1], points[:,2])

  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')

  # same scale for all axis
  ax.set_aspect('equal', adjustable='box')

  plt.show()


def cluster_points(points):

  # grouped_points contains all points with the same label (labeled by the authors)
  # however, they did not differentiate between instances
  # we therefore cluster these points with a distance of 1m and conclude each cluster to be an instance

  # https://stackoverflow.com/questions/56062673/clustering-the-3d-points-when-given-the-x-y-z-coordinates-using-dbscan-algorithm
  model = DBSCAN(eps=1, min_samples=2) # cluster points in a range of 1m, require two points to consider point as a cluster core
  model.fit_predict(points)
  pred = model.fit_predict(points)

  n_discovered_clusters = len(set(model.labels_))

  if n_discovered_clusters >= 2:

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], c=model.labels_)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # same scale for all axis
    ax.set_aspect('equal', adjustable='box')
    ax.view_init(azim=200)

    plt.show()

  print("number of cluster found: {}".format(n_discovered_clusters))
  print('cluster for each point: ', model.labels_)


def filter_by_label_id_and_groupd_by_label_and_instance(points, labels, label_ids_to_select=set() ):

  label_ids = labels & 0xFFFF # lower half
  instance_ids = labels >> 16 # upper half

  # grouped_points[<label_id>][<instance_id>] = [<point0>, <point1>, ...]
  grouped_points = defaultdict(lambda: defaultdict(list))

  for point, label_id, instance_id in zip(points, label_ids, instance_ids):

    if label_id in label_ids_to_select:
      grouped_points[label_id][instance_id].append(point)

  # print(f'selected scans/labels: {len(selected_scans)}')

  return grouped_points


def group_points_by_label_and_instance(points, labels):
  # collect instances in this dict
  # key: label, value: [np.array[coordinates]]
  # example {42: ('person', [239.34, 172.3, -12, 1])}
  points_grouped_by_label = defaultdict(list)

  # group scans with the same label (instance id + semantic id)
  for point, label in zip(points, labels):
    points_grouped_by_label[label].append(point)

  return [ # as list with (points, labels) tuples
    (np.array(points, dtype=np.float32), np.array(label))
    for label, points in points_grouped_by_label.items()
  ]


def object_movement_history(points, label):

  points_grouped_by_label = defaultdict(list)

  for point, label in zip(points, label):
    points_grouped_by_label[label].append(point)

  point_stats_by_label = defaultdict(list)

  for label, points in points_grouped_by_label.items():
    point_bounding_box_lowest, point_bounding_box_highest, point_center = generate_bounding_box_and_center_points(points)
    point_stats_by_label[label] = {
      'point_bounding_box_lowest': point_bounding_box_lowest,
      'point_bounding_box_highest': point_bounding_box_highest,
      'point_center': point_center
    }



def generate_bounding_box_and_center_points(points):

  highest_x, highest_y, highest_z = float("-inf"), float("-inf"), float("-inf")
  lowest_x, lowest_y, lowest_z = float("inf"), float("inf"), float("inf")

  x_sum = 0
  y_sum = 0
  z_sum = 0

  # for only these with highest x-, y-, and z-coordinates
  for x, y, z, _ in points:

    x_sum += x
    y_sum += y
    z_sum += z

    if x > highest_x:
      highest_x = x
    if x < lowest_x: # not elif for the case len(points) == 1
      lowest_x = x

    if y > highest_y:
      highest_y = y
    if y < lowest_y: # not elif for the case len(points) == 1
      lowest_y = y
    
    if z > highest_z:
      highest_z = z
    if z < lowest_z: # not elif for the case len(points) == 1
      lowest_z = z

    # set lowest and highest point that span the bounding-box
    point_bounding_box_lowest = np.array([
      lowest_x,
      lowest_y,
      lowest_z
    ], dtype=np.float32)

    point_bounding_box_highest = np.array([
      highest_x,
      highest_y,
      highest_z
    ], dtype=np.float32)

    # calculate center-point by averaging each coordinate dimension
    point_center = np.array([
      x_sum / len(points),
      y_sum / len(points),
      z_sum / len(points)
    ], dtype=np.float32)
  
  return point_bounding_box_lowest, point_bounding_box_highest, point_center


if __name__ == '__main__':
  start_time = time.time()

  parser = argparse.ArgumentParser("./generate_global_instance_bounding_boxes.py")

  parser.add_argument(
      '--dataset',
      '-d',
      type=str,
      required=True,
      help='dataset folder containing all sequences in a folder called "sequences".',
  )

  parser.add_argument(
      '--output',
      '-o',
      type=str,
      required=True,
      help='output folder for generated sequence scans.',
  )

  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("*" * 80)
  print(" dataset folder: ", FLAGS.dataset)
  print("  output folder: ", FLAGS.output)
  print("*" * 80)


  sequences_dir = os.path.join(FLAGS.dataset, "sequences")
  sequence_folders = [
      f for f in sorted(os.listdir(sequences_dir))
      if os.path.isdir(os.path.join(sequences_dir, f))
  ]

  # iterate all sequences eg. 00, 01, etc.
  for folder in sequence_folders:
    # print('TODO: skipping first sequence folder') # TODO!!!

    input_folder = os.path.join(sequences_dir, folder)
    output_folder = os.path.join(FLAGS.output, "sequences", folder)
    velodyne_folder = os.path.join(output_folder, "velodyne")
    labels_folder = os.path.join(output_folder, "labels")

    if os.path.exists(output_folder) or os.path.exists(
            velodyne_folder) or os.path.exists(labels_folder):
      print("Output folder '{}' already exists!".format(output_folder))
      answer = 'y' #input("Overwrite? [y/N] ") # TODO: change back
      if answer != "y":
        print("Aborted.")
        exit(1)
      if not os.path.exists(velodyne_folder):
        os.makedirs(velodyne_folder)
      if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)
    else:
      os.makedirs(velodyne_folder)
      os.makedirs(labels_folder)

    shutil.copy(os.path.join(input_folder, "poses.txt"), output_folder)
    shutil.copy(os.path.join(input_folder, "calib.txt"), output_folder)

    scan_files = [
        f for f in sorted(os.listdir(os.path.join(input_folder, "velodyne")))
        if f.endswith(".bin")
    ]

    calibration = parse_calibration(os.path.join(input_folder, "calib.txt"))
    poses = parse_poses(os.path.join(input_folder, "poses.txt"), calibration)
    base_pose_inv = inv(poses[0])

    # stores the inter-frame movement history of each object (center-points)
    # dict[<label_id>][<instance-id>] = [<center-point0>, <center-point1>, ...]
    point_center_history_by_label_id_and_sequence_id = defaultdict(lambda: defaultdict(list))    

    # iterate frames/scans
    for frame_id, file in enumerate(pbar := tqdm(scan_files)):
      pbar.set_description(f"Processing {folder}/{file}")
      
      # read scan and labels, get pose
      scan_filename = os.path.join(input_folder, "velodyne", file)
      points = np.fromfile(scan_filename, dtype=np.float32).reshape((-1, 4))

      label_filename = os.path.join(input_folder, "labels", os.path.splitext(file)[0] + ".label")
      labels = np.fromfile(label_filename, dtype=np.uint32).reshape((-1))

      points = translate_points_to_pose(base_pose_inv=base_pose_inv, current_pose=poses[frame_id], points_original=points)

      points_grouped_by_label_id_and_instance_id = filter_by_label_id_and_groupd_by_label_and_instance(points, labels, label_ids_to_select={30})
      for label_id, points_grouped_by_instance_id in points_grouped_by_label_id_and_instance_id.items():
        for instance_id, points in points_grouped_by_instance_id.items():

          point_bb_low, point_bb_high, point_center = generate_bounding_box_and_center_points(points)

          span_bb = point_bb_high - point_bb_low
          span_volume_bb = np.prod(span_bb[:3])

          point_center_history_by_label_id_and_sequence_id[label_id][instance_id].append(point_center)

          # if span_volume_bb >= 0.1:
          #   print(f'span_volume_bb={span_volume_bb}')
          #   print(f'instance_id={instance_id}')
          #   points = np.array(points, dtype=np.float32)
          #   plot_points_3d(points)

        # print(f'grouped_points={grouped_points[:, 0]}, grouped_points_translated={grouped_points_translated[:, 0]}')

      # scan_bb, labels_bb, scan_cp, label_cp = generate_bounding_boxes_and_center_points(scans, labels)
      # # add bounding-box points and labels to the existing arrays
      # scans = np.concatenate((scans, bb_scans))
      # labels = np.concatenate((labels, bb_labels))

      # points.tofile(os.path.join(velodyne_folder, file))
      # labels.tofile(os.path.join(labels_folder, os.path.splitext(file)[0] + ".label")) 

    # calculate movement stats per instance
    for label_id, points_center_by_instance_id in point_center_history_by_label_id_and_sequence_id.items():
      for instance_id, points_center in points_center_by_instance_id.items():
        points_center = np.array(points_center)
        points_center.tofile(f'object_traces/sequence={folder}_label={label_id:03d}_instance={instance_id:03d}.np')


  print("execution time: {}".format(time.time() - start_time))