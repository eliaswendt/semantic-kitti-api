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

    translated_pose = np.matmul(Tr_inv, np.matmul(pose, Tr))
    poses.append(translated_pose)

  return poses


def generate_bounding_boxes(points, labels):

  instance_ids = labels >> 16      # upper half
  label_ids = labels & 0xFFFF   # lower half

  # collect instances in this dict
  # key: label, value: [np.array[coordinates]]
  # example {42: ('person', [239.34, 172.3, -12, 1])}
  points_grouped_by_label = defaultdict(list)

  # group points with the same labels
  # we use the entire label (instance_id + label_id) as key
  for i in range(len(points)):
    if label_ids[i] != 0: # skip unlabeled
      points_grouped_by_label[labels[i]].append(points[i])

  
  bb_points = np.zeros((len(points_grouped_by_label) * 2, 4))
  bb_labels = np.zeros((len(points_grouped_by_label) * 2,))


  # we will later use these extreme coordinates to compose the instance's bounding box
  i = 0
  for points in points_grouped_by_label.values():
    highest_x, highest_y, highest_z = float("-inf"), float("-inf"), float("-inf")
    lowest_x, lowest_y, lowest_z = float("inf"), float("inf"), float("inf")

    # iterate all points of instance and filter
    # for only these with highest x-, y-, and z-coordinates
    for x, y, z, _ in points:
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
    bb_points[i] = np.array([highest_x, highest_y, highest_z, 1.])
    bb_labels[i] = np.array([2]) # unused label id
    bb_points[i+1] = np.array([lowest_x, lowest_y, lowest_z, 1.], dtype=np.float32)
    bb_labels[i+1] = np.array([2]) # unused label id
  
    i += 2

  return bb_points, bb_labels


if __name__ == '__main__':
  start_time = time.time()

  parser = argparse.ArgumentParser("./generate_sequential.py")

  parser.add_argument(
      '--datacfg', '-dc',
      type=str,
      required=False,
      default="config/semantic-kitti.yaml",
      help='Dataset config file. Defaults to %(default)s',
  )

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

  parser.add_argument(
      '--sequence_length',
      '-s',
      type=int,
      required=True,
      help='length of sequence, i.e., how many scans are concatenated.',
  )

  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("*" * 80)
  print(" datacfg folder: ", FLAGS.datacfg)
  print(" dataset folder: ", FLAGS.dataset)
  print("  output folder: ", FLAGS.output)
  print("sequence length: ", FLAGS.sequence_length)
  print("*" * 80)

  print("Opening data config file %s" % FLAGS.datacfg)
  DATA = yaml.safe_load(open(FLAGS.datacfg, 'r'))

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
      answer = input("Overwrite? [y/N] ")
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


    print("Processing {} ".format(folder), end="", flush=True)

    for i, f in enumerate(scan_files[:1]):
      print(f'processing file "{f}"')
      # read scan and labels, get pose
      scan_filename = os.path.join(input_folder, "velodyne", f)
      scans = np.fromfile(scan_filename, dtype=np.float32).reshape((-1, 4))

      label_filename = os.path.join(input_folder, "labels", os.path.splitext(f)[0] + ".label")
      labels = np.fromfile(label_filename, dtype=np.uint32).reshape((-1))

      bb_scans, bb_labels = generate_bounding_boxes(points=scans, labels=labels)
      
      # add bounding-box points and labels to the existing arrays
      scans = np.concatenate((scans, bb_scans))
      labels = np.concatenate((labels, bb_labels))

      scans.tofile(os.path.join(velodyne_folder, f))
      labels.tofile(os.path.join(labels_folder, os.path.splitext(f)[0] + ".label")) 



    # history = deque()

    # progress = 10

    #   pose = poses[i]

    #   # prepare single numpy array for all points that can be written at once.
    #   num_concat_points = points.shape[0]
    #   num_concat_points += sum([past["points"].shape[0] for past in history])
    #   concated_points = np.zeros((num_concat_points * 4), dtype = np.float32)
    #   concated_labels = np.zeros((num_concat_points), dtype = np.uint32)

    #   start = 0
    #   concated_points[4 * start:4 * (start + points.shape[0])] = scan.reshape((-1))
    #   concated_labels[start:start + points.shape[0]] = labels
    #   start += points.shape[0]

    #   for past in history:
    #     diff = np.matmul(inv(pose), past["pose"])
    #     tpoints = np.matmul(diff, past["points"].T).T
    #     tpoints[:, 3] = past["remissions"]
    #     tpoints = tpoints.reshape((-1))

    #     concated_points[4 * start:4 * (start + past["points"].shape[0])] = tpoints
    #     concated_labels[start:start + past["labels"].shape[0]] = past["labels"]
    #     start += past["points"].shape[0]


    #   # write scan and labels in one pass.
    #   concated_points.tofile(os.path.join(velodyne_folder, f))
    #   concated_labels.tofile(os.path.join(labels_folder, os.path.splitext(f)[0] + ".label")) 

    #   # append current data to history queue.
    #   history.appendleft({
    #       "points": points,
    #       "labels": labels,
    #       "remissions": remissions,
    #       "pose": pose.copy()
    #   })

    #   if len(history) >= FLAGS.sequence_length:
    #     history.pop()

    #   if 100.0 * i / len(scan_files) >= progress:
    #     print(".", end="", flush=True)
    #     progress = progress + 10
    # print("finished.")


  print("execution time: {}".format(time.time() - start_time))