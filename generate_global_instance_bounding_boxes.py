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


def transform_points_from_velo_to_cam_2_coordinate_system(points):
  """
  In order to transform a homogeneous point X = [x y z 1]' from the velodyne
  coordinate system to a homogeneous point Y = [u v 1]' on image plane of
  camera xx, the following transformation has to be applied:

  Y = P_rect_xx * R_rect_00 * (R|T)_velo_to_cam * X
  """

  # calib_velo_to_cam.txt and calib_cam_to_cam downloaded from: https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26/calib_cam_to_cam.txt and https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26/calib_velo_to_cam.txt
  # however: not sure if these values apply to our dataset (semantic kitti aka. kitti odometry)

  # compose of R and T from calib_velo_to_cam.txt
  RT_velo_to_cam = np.array([
    [7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
    [1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],
    [9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01],
    [0.,            0,             0,             1.          ]
  ], dtype=np.float32)
  # RT_velo_to_cam = cam_calibration['TR'].reshape((3,4))
  # RT_velo_to_cam = np.zeros((4, 4), dtype=np.float32)
  # RT_velo_to_cam[0, 0:4] = cam_calibration['TR'][0:4]
  # RT_velo_to_cam[1, 0:4] = cam_calibration['TR'][4:8]
  # RT_velo_to_cam[2, 0:4] = cam_calibration['TR'][8:12]
  # RT_velo_to_cam[3, 3] = 1.0


  # R_rect_00 from calib_cam_to_cam.txt: 9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 -4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01
  R_rect_00 = np.array([
    [ 9.999239e-01, 9.837760e-03, -7.445048e-03, 0.],
    [-9.869795e-03, 9.999421e-01, -4.278459e-03, 0.],
    [ 7.402527e-03, 4.351614e-03,  9.999631e-01, 0.],
    [ 0.,           0.,             0.,          1.]
  ], dtype=np.float32)
  # R_rect_00 = cam_calibration['R_rect_00'].reshape((3,3))
  # R_rect_00 = np.zeros((4, 4), dtype=np.float32)
  # R_rect_00[0, 0:3] = cam_calibration['R_rect_00'][0:3]
  # R_rect_00[1, 0:3] = cam_calibration['R_rect_00'][3:6]
  # R_rect_00[2, 0:3] = cam_calibration['R_rect_00'][6:9]
  # R_rect_00[3, 3] = 1.0



  # P_rect_02 from calib_cam_to_cam.txt: 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03
  P_rect_02 = np.array([
    [7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
    [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
    [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03],
  ], dtype=np.float32)
  # P_rect_02 = cam_calibration['P_rect_02'].reshape((3, 4))
  # P_rect_02 = np.zeros((4,4), dtype=np.float32)
  # P_rect_02[0, 0:4] = cam_calibration['P_rect_02'][0:4]
  # P_rect_02[1, 0:4] = cam_calibration['P_rect_02'][4:8]
  # P_rect_02[2, 0:4] = cam_calibration['P_rect_02'][8:12]
  # P_rect_02[3, 3] = 1.0



  # Tr from calib.txt: 4.276802385584e-04 -9.999672484946e-01 -8.084491683471e-03 -1.198459927713e-02 -7.210626507497e-03 8.081198471645e-03 -9.999413164504e-01 -5.403984729748e-02 9.999738645903e-01 4.859485810390e-04 -7.206933692422e-03 -2.921968648686e-01
  Tr = np.array([
    [4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02],
    [-7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02],
    [9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01],
    [0., 0., 0., 1.]
  ], dtype=np.float32)

  # P0 from calib.txt: 7.070912000000e+02 0.000000000000e+00 6.018873000000e+02 0.000000000000e+00 0.000000000000e+00 7.070912000000e+02 1.831104000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
  P0 = np.array([
    [7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02, 0.000000000000e+00],
    [0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02, 0.000000000000e+00],
    [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00],
    [0., 0., 0., 1.]
  ], dtype=np.float32)

  # P2 from calib.txt: 7.070912000000e+02 0.000000000000e+00 6.018873000000e+02 4.688783000000e+01 0.000000000000e+00 7.070912000000e+02 1.831104000000e+02 1.178601000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 6.203223000000e-03
  P2 = np.array([
    [7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02, 4.688783000000e+01],
    [0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02, 1.178601000000e-01],
    [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 6.203223000000e-03],
  ], dtype=np.float32)

  # save remissions before point transformation
  remissions = points[:, 3]

  # set neutral element
  points[:, 3] = 1.
  
  # transpose for batched matmul
  points_translated = points.T

  # transformations as described in https://github.com/bostondiditeam/kitti/blob/master/resources/devkit_object/readme.txt#L87
  # points_translated = np.matmul(RT_velo_to_cam, points_translated)
  # points_translated = np.matmul(R_rect_00, points_translated)
  # points_translated = np.matmul(P2, points_translated)

  # transformations as described in https://github.com/yfcube/kitti-devkit-odom/blob/master/readme.txt#L91
  # points_translated = np.matmul(Tr, points_translated)
  # points_translated = np.matmul(P0, points_translated)
  # points_translated = np.matmul(P2, points_translated)


  # rot_mat = np.matmul(P2, Tr)
  # points_translated = np.matmul(rot_mat, points_translated)

  # revert transpose
  points_translated = points_translated.T

  # additional transformations from https://github.com/valeoai/xmuda/blob/master/xmuda/data/semantic_kitti/preprocess.py#L108
  points_translated = points_translated[:, :2] / np.expand_dims(points_translated[:, 2], axis=1)  # scale 2D points
  points_translated = np.fliplr(points_translated)

  # re-built 4d points-array to stay compatible with visualizer.py
  # only necessary if points transformation yields non-4d point coordinates
  points_translated_4d = np.zeros((points_translated.shape[0], 4), dtype=np.float32)
  points_translated_4d[:, 0:2] = points_translated[:, 0:2]
  points_translated_4d[:, 3] = remissions # re-set remissions

  return points_translated_4d


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


def transform_points_from_velo_to_pose_coordinate_system(base_pose_inv, current_pose, points_original):
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


def plot_points_2d(points):
  # in case scan.shape is (,2)
  points = points.reshape((-1, 2))

  fig = plt.figure()
  ax = fig.add_subplot()
  ax.scatter(points[:,0], points[:,1])

  ax.set_xlabel('x')
  ax.set_ylabel('y')

  # same scale for all axis
  ax.set_aspect('equal', adjustable='box')

  plt.show()


def cluster_points(points):
  # currently unused, as dataset we work with is already labeled with instance-ids
  # we cluster points with a distance of 1m and conclude each cluster to be an instance

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


def filter_by_label_id_then_group_by_label_id_and_instance_id(points, labels, label_ids_to_select=set()):

  # label-ids are listed here: https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml#L2

  label_ids = labels & 0xFFFF # lower half
  instance_ids = labels >> 16 # upper half

  # grouped_points[<label_id>][<instance_id>] = [<point0>, <point1>, ...]
  grouped_points = defaultdict(lambda: defaultdict(list))

  for point, label_id, instance_id in zip(points, label_ids, instance_ids):

    if label_id in label_ids_to_select:
      grouped_points[label_id][instance_id].append(point)

  # print(f'selected scans/labels: {len(selected_scans)}')

  return grouped_points


def generate_bbox_and_center_points(points):

  highest_x, highest_y, highest_z = float("-inf"), float("-inf"), float("-inf")
  lowest_x, lowest_y, lowest_z = float("inf"), float("inf"), float("inf")
  x_sum, y_sum, z_sum = 0, 0, 0

  # for only these with highest x-, y-, and z-coordinates
  for x, y, z, _ in points:

    x_sum += x
    y_sum += y
    z_sum += z

    if x > highest_x:
      highest_x = x
    if x < lowest_x: # not elif in case len(points) == 1
      lowest_x = x

    if y > highest_y:
      highest_y = y
    if y < lowest_y: # not elif in case len(points) == 1
      lowest_y = y
    
    if z > highest_z:
      highest_z = z
    if z < lowest_z: # not elif in case len(points) == 1
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

  #cam_calibration = parse_cam_calibration(os.path.join(FLAGS.dataset, "sequences/calib_cam.txt"))

  sequences_dir = os.path.join(FLAGS.dataset, "sequences")
  sequence_folders = [
      f for f in sorted(os.listdir(sequences_dir))
      if os.path.isdir(os.path.join(sequences_dir, f))
  ]

  # iterate all sequences eg. 00, 01, etc.
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

      img_filename = os.path.join(input_folder, "image_2", os.path.splitext(file)[0] + ".png")
      image = plt.imread(img_filename)

      points_in_camera_coordinates = transform_points_from_velo_to_cam_2_coordinate_system(points)

      # plot_points_2d(points_in_camera_coordinates[:, :2])

      # filter transformed points array to fit inside the camera image plane
      # filtered_points_in_camera_coordinates = []
      # filtered_labels = []
      # max_x = image.shape[0]
      # max_y = image.shape[1]
      # for point, label in zip(points_in_camera_coordinates, labels):
      #   y, x = int(point[0]), int(point[1])

      #   if x >= 0. and x < max_x and y >= 0. and y < max_y:
      #     image[x][y][:] = [1., 0., 0.]
      #     filtered_points_in_camera_coordinates.append(point)
      #     filtered_labels.append(label)

      # #plot_points_2d(points_in_camera_coordinates[:, :2])
      # plt.imshow(image)
      # plt.show()

      # points = translate_points_to_pose(base_pose_inv=base_pose_inv, current_pose=poses[frame_id], points_original=points)
      # continue

      points_grouped_by_label_id_and_instance_id = filter_by_label_id_then_group_by_label_id_and_instance_id(
        points, 
        labels, 
        label_ids_to_select={30, 31, 253, 254} # label-ids are listed here: https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml#L2
      )

      for label_id, points_grouped_by_instance_id in points_grouped_by_label_id_and_instance_id.items():
        for instance_id, points in points_grouped_by_instance_id.items():

          point_bbox_low, point_bbox_high, point_center = generate_bbox_and_center_points(points)

          # calculate bbox volume
          bbox_span = point_bbox_high - point_bbox_low
          bbox_volume = np.prod(bbox_span[:3])

          point_center_history_by_label_id_and_sequence_id[label_id][instance_id].append(point_center)

          # if span_volume_bb >= 0.1:
          #   print(f'span_volume_bb={span_volume_bb}')
          #   print(f'instance_id={instance_id}')
          #   points = np.array(points, dtype=np.float32)
          #   plot_points_3d(points)

        # print(f'grouped_points={grouped_points[:, 0]}, grouped_points_translated={grouped_points_translated[:, 0]}')

      # TODO: enable to save transformed points
      # points.tofile(os.path.join(velodyne_folder, file))
      # labels.tofile(os.path.join(labels_folder, os.path.splitext(file)[0] + ".label")) 

    # save instance traces to file to get analyzed `object_trace_analysis.ipynb`
    # for label_id, points_center_by_instance_id in point_center_history_by_label_id_and_sequence_id.items():
    #   for instance_id, points_center in points_center_by_instance_id.items():
    #     points_center = np.array(points_center)
    #     points_center.tofile(f'object_traces/sequence={folder}_label={label_id:03d}_instance={instance_id:03d}.np')


  print("execution time: {}".format(time.time() - start_time))