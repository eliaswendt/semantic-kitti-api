import argparse
import os
import shutil
import time
import numpy as np
from collections import defaultdict
from numpy.linalg import inv

import yaml

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

def translate_scan_to_pose(base_pose_inv, current_pose, scan):
  # transform from current_pose to base_pose
  current_to_base = np.matmul(base_pose_inv, current_pose)

  points = np.ones((scan.shape[0], 4))
  points[:, 0:3] = scan[:, 0:3]
  remissions = scan[:, 3]

  # apply pose transformation to points
  points = np.matmul(current_to_base, points.T).T
  points[:, 3] = remissions # re-add remissions

  return points.astype(np.float32)


def filter_label_ids(points, labels, filtered_label_ids=[]):

  label_ids = labels & 0xFFFF   # lower half

  filtered_points = []
  filtered_labels = []

  filter_count = 0

  for i in range(len(points)):
    if label_ids[i] in filtered_label_ids:
      filter_count += 1
    else:
      filtered_points.append(points[i])
      filtered_labels.append(labels[i])

  print(f'filtered scans/labels: {filter_count}')

  return np.array(filtered_points, dtype=np.float32), np.array(filtered_labels, dtype=np.uint32)


def generate_bounding_boxes(scans, labels):

  instance_ids = labels >> 16      # upper half
  label_ids = labels & 0xFFFF   # lower half

  # collect instances in this dict
  # key: label, value: [np.array[coordinates]]
  # example {42: ('person', [239.34, 172.3, -12, 1])}
  scans_grouped_by_instance = defaultdict(list)

  # group scans with the same label (instance id + semantic id)
  for scan, label in zip(scans, labels):
    scans_grouped_by_instance[label].append(scan)

  bb_scans = []
  bb_labels = []

  # we will later use these extreme coordinates to compose the instance's bounding box
  for label, grouped_scans in scans_grouped_by_instance.items():

    if len(grouped_scans) < 2:
      # ignoring instances with only one point
      continue

    highest_x, highest_y, highest_z = float("-inf"), float("-inf"), float("-inf")
    lowest_x, lowest_y, lowest_z = float("inf"), float("inf"), float("inf")

    # iterate all points of instance and filter
    # for only these with highest x-, y-, and z-coordinates
    for x, y, z, _ in grouped_scans:
      if x > highest_x:
        highest_x = x
      if x < lowest_x:
        lowest_x = x

      if y > highest_y:
        highest_y = y
      if y < lowest_y:
        lowest_y = y
      
      if z > highest_z:
        highest_z = z
      if z < lowest_z:
        lowest_z = z

    # create and insert points that span the instance's bounding box
    bb_scans.append([highest_x, highest_y, highest_z, 99.]) # set max remission
    bb_scans.append([lowest_x, lowest_y, lowest_z, 99.]) # set max remission
    bb_labels.append(label)
    bb_labels.append(label)
  
  return np.array(bb_scans, dtype=np.float32), np.array(bb_labels, dtype=np.uint32)


if __name__ == '__main__':
  start_time = time.time()

  parser = argparse.ArgumentParser("./generate_sequential.py")

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

  for folder in sequence_folders:
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

    for i, f in enumerate(scan_files[:100]):
      print(f'Processing {folder}/{f}')

      # read scan and labels, get pose
      scan_filename = os.path.join(input_folder, "velodyne", f)
      scan = np.fromfile(scan_filename, dtype=np.float32).reshape((-1, 4))

      label_filename = os.path.join(input_folder, "labels", os.path.splitext(f)[0] + ".label")
      labels = np.fromfile(label_filename, dtype=np.uint32).reshape((-1))

      # filtered = filter_label_ids(scans, labels, filtered_label_ids=[0])
      # bb_scans, bb_labels = generate_bounding_boxes(*filtered)
      
      # # add bounding-box points and labels to the existing arrays
      # scans = np.concatenate((scans, bb_scans))
      # labels = np.concatenate((labels, bb_labels))

      scan = translate_scan_to_pose(base_pose_inv=base_pose_inv, current_pose=poses[i], scan=scan)

      print(f'{scan.shape[0]}|{labels.shape[0]}')

      scan.tofile(os.path.join(velodyne_folder, f))
      labels.tofile(os.path.join(labels_folder, os.path.splitext(f)[0] + ".label")) 


  print("execution time: {}".format(time.time() - start_time))