import numpy as np
import math
from scipy.spatial import cKDTree

def _pc_normalize(pc):
    """
    Normalize the point cloud to a unit sphere
    :param pc: input point cloud
    :return: normalized point cloud
    """
    centroid = np.mean(pc[:, :3], axis=0)
    pc[:, :3] = pc[:, :3] - centroid
    m = np.max(np.sqrt(np.sum(pc[:, :3] ** 2, axis=1)))
    pc[:, :3] = pc[:, :3] / m
    return pc


def _shuffle_pointcloud(pcd):
    """
    Shuffle the points
    :param pcd: input point cloud
    :return: shuffled point clouds
    """
    idx = np.random.rand(pcd.shape[0], 1).argsort(axis=0)
    return np.take_along_axis(pcd, idx, axis=0)


def _gen_random_cluster_sizes(num_clusters, total_cluster_size):
    """
    Generate random cluster sizes
    :param num_clusters: number of clusters
    :param total_cluster_size: total size of all clusters
    :return: a list of each cluster size
    """
    rand_list = np.random.randint(num_clusters, size=total_cluster_size)
    cluster_size_list = [sum(rand_list == i) for i in range(num_clusters)]
    return cluster_size_list


def _sample_points_inside_unit_sphere(number_of_particles):
    """
    Uniformly sample points in a unit sphere
    :param number_of_particles: number of points to sample
    :return: sampled points
    """
    radius = np.random.uniform(0.0, 1.0, (number_of_particles, 1))
    radius = np.power(radius, 1 / 3)
    costheta = np.random.uniform(-1.0, 1.0, (number_of_particles, 1))
    theta = np.arccos(costheta)
    phi = np.random.uniform(0, 2 * np.pi, (number_of_particles, 1))
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.concatenate([x, y, z], axis=1)


def corrupt_scale(pointcloud, level):
    """
    Corrupt the scale of input point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    s = [1.6, 1.7, 1.8, 1.9, 2.0][level]
    xyz = np.random.uniform(low=1. / s, high=s, size=[3])
    pointcloud[:,:3] = _pc_normalize(np.multiply(pointcloud[:,:3], xyz).astype('float32'))
    return pointcloud


def corrupt_jitter(pointcloud, level):
    """
    Jitter the input point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    sigma = 0.01 * (level + 1)
    N, C = pointcloud[:, :3].shape
    pointcloud[:, :3] = pointcloud[:, :3] + sigma * np.random.randn(N, C)
    return pointcloud


def corrupt_rotate(pointcloud, level):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    angle_clip = math.pi / 6
    angle_clip = angle_clip / 5 * (level + 1)
    angles = np.random.uniform(-angle_clip, angle_clip, size=(3))
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    pointcloud[:, :3] = np.dot(pointcloud[:, :3], R)
    return pointcloud


def corrupt_dropout_global(pointcloud, level):
    """
    Drop random points globally
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    drop_rate = [0.25, 0.375, 0.5, 0.625, 0.75][level]
    num_points = pointcloud.shape[0]
    pointcloud = _shuffle_pointcloud(pointcloud)
    pointcloud = pointcloud[:int(num_points * (1 - drop_rate)), :]
    return pointcloud


def corrupt_dropout_local(pointcloud, level):
    """
    Randomly drop local clusters
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    num_points = pointcloud.shape[0]
    total_cluster_size = 100 * (level + 1)
    num_clusters = np.random.randint(1, 8)
    cluster_size_list = _gen_random_cluster_sizes(num_clusters, total_cluster_size)
    for i in range(num_clusters):
        K = cluster_size_list[i]
        pointcloud = _shuffle_pointcloud(pointcloud)
        dist = np.sum((pointcloud[:,:3] - pointcloud[:1, :3]) ** 2, axis=1, keepdims=True)
        idx = dist.argsort(axis=0)[::-1, :]
        pointcloud = np.take_along_axis(pointcloud, idx, axis=0)
        num_points -= K
        pointcloud = pointcloud[:num_points, :]
    return pointcloud


def corrupt_add_global(pointcloud, level):
    """
    Add random points globally
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    npoints = 10 * (level + 1)
    additional_pointcloud = _sample_points_inside_unit_sphere(npoints)
    additional_pointcloud = get_color(additional_pointcloud, pointcloud)    # add color
    pointcloud = np.concatenate([pointcloud, additional_pointcloud[:npoints]], axis=0)
    return pointcloud


def get_color(add_pcd, pointcloud):
    """
    Get color for added points
    :param add_pcd: added points
    :param pointcloud: original point cloud
    :return: added points with color
    """

    pointcloud_coords = pointcloud[:, :3]
    pointcloud_colors = pointcloud[:, 3:]

    tree = cKDTree(pointcloud_coords)

    _, indices = tree.query(add_pcd)

    add_pcd_colors = pointcloud_colors[indices]

    add_pcd_with_colors = np.concatenate([add_pcd, add_pcd_colors], axis=1)
    return add_pcd_with_colors


def corrupt_add_local(pointcloud, level):
    """
    Randomly add local clusters to a point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    num_points = pointcloud.shape[0]
    total_cluster_size = 100 * (level + 1)
    num_clusters = np.random.randint(1, 8)
    cluster_size_list = _gen_random_cluster_sizes(num_clusters, total_cluster_size)
    pointcloud = _shuffle_pointcloud(pointcloud)
    add_pcd = np.zeros_like(pointcloud[:, :3])
    num_added = 0
    for i in range(num_clusters):
        K = cluster_size_list[i]
        sigma = np.random.uniform(0.075, 0.125)
        add_pcd[num_added:num_added + K, :] = np.copy(pointcloud[i:i + 1, :3])
        add_pcd[num_added:num_added + K, :] = add_pcd[num_added:num_added + K, :] + sigma * np.random.randn(
            *add_pcd[num_added:num_added + K, :].shape)
        num_added += K
    assert num_added == total_cluster_size
    dist = np.sum(add_pcd ** 2, axis=1, keepdims=True).repeat(3, axis=1)
    add_pcd[dist > 1] = add_pcd[dist > 1] / dist[dist > 1]  # ensure the added points are inside a unit sphere
    add_pcd = get_color(add_pcd, pointcloud)
    pointcloud = np.concatenate([pointcloud, add_pcd], axis=0)
    pointcloud = pointcloud[:num_points + total_cluster_size]
    return pointcloud


if __name__ == '__main__':
    # test code
    from all_tools import read_ply2np,save_ply_from_np

    pointcloud = read_ply2np(r"ply_path")[:, :6]
    pointcloud = _pc_normalize(pointcloud)
    pointcloud_c = corrupt_dropout_local(pointcloud, 4)
    save_ply_from_np(pointcloud_c, "path_to_save")
