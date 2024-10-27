import os
import numpy as np
from sklearn.model_selection import train_test_split
from plyfile import PlyElement, PlyData

def read_ply(filename):
    # 以二进制模式打开 PLY 文件
    with open(filename, 'rb') as f:
        ply_data = PlyData.read(f)

    # 获取顶点信息
    vertex_data = ply_data['vertex']
    properties = vertex_data.properties

    # 获取属性名
    property_names = [prop.name for prop in properties]

    # 提取数据并转换为 numpy 数组
    extracted_data = []
    for name in property_names:
        extracted_data.append(np.array(vertex_data[name]))

    # 转置为二维数组
    data_matrix = np.vstack(extracted_data).T

    return data_matrix


# 保存点云到txt文件
def save_points(points, filename):
    np.savetxt(filename, points, fmt='%f')


# 保存实例分割到单独的txt文件
def save_instance_segmentation(points, folder):
    labels = np.unique(points[:, -1])
    for label in labels:
        instance_points = points[points[:, -1] == label]
        if label == 0:
            instance_filename = os.path.join(folder, f'ceiling_1.txt')
        else:
            instance_filename = os.path.join(folder, f'floor_{int(label)}.txt')
        save_points(instance_points[:,:6], instance_filename)

def get_files_from_txt(txt_path):
    with open(txt_path, 'r') as file:
        lines = file.readlines()
    # Remove "data/" and strip newlines
    return [line.replace('data/Crops3D/', '').strip().replace('\\', '/') for line in lines if line.strip()]






def process_dataset(path, output_folder):

    os.makedirs(output_folder, exist_ok=True)


    # train_files, test_files = train_test_split(ply_files, test_size=0.3, random_state=42)
    #read train_files, test_files
    # path = "Crop3D_IS"
    train_files = get_files_from_txt(os.path.join(path, 'train.txt'))
    test_files = get_files_from_txt(os.path.join(path, 'test.txt'))


    area1_folder = os.path.join(output_folder, 'Area_1')
    area5_folder = os.path.join(output_folder, 'Area_5')
    os.makedirs(area1_folder, exist_ok=True)
    os.makedirs(area5_folder, exist_ok=True)
    for train_file in train_files:
        print(train_file)
        points = read_ply(train_file)
        ply_name = os.path.splitext(os.path.basename(train_file))[0]
        ply_folder = os.path.join(area1_folder, ply_name)
        os.makedirs(ply_folder, exist_ok=True)
        save_points(points[:,:6], os.path.join(ply_folder, f'{ply_name}.txt'))
        instance_folder = os.path.join(ply_folder, 'Annotations')
        os.makedirs(instance_folder, exist_ok=True)
        save_instance_segmentation(points, instance_folder)


    for test_file in test_files:
        print(test_file)
        points = read_ply(test_file)
        ply_name = os.path.splitext(os.path.basename(test_file))[0]
        ply_folder = os.path.join(area5_folder, ply_name)
        os.makedirs(ply_folder, exist_ok=True)
        save_points(points[:,:6], os.path.join(ply_folder, f'{ply_name}.txt'))
        instance_folder = os.path.join(ply_folder, 'Annotations')
        os.makedirs(instance_folder, exist_ok=True)
        save_instance_segmentation(points, instance_folder)



# global path
path = "Crops3D_IS"
# ply_files = [file for file in os.listdir(path) if file.endswith('.ply')]

process_dataset(path, 'output_dataset')