if 1:
    # natural sort
    !pip
    install
    natsort

kaggle_dir = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification'

import matplotlib
import matplotlib.pyplot as plt

import glob
import pydicom
from natsort import natsorted, ns
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


# disable SettingWithCopyWarning


# helper
class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


print('import ok')


def normalise_to_8bit(x, lower=0.1, upper=99.9):  # 1, 99 #0.05, 99.5 #0, 100
    lower, upper = np.percentile(x, (lower, upper))
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    x = x / np.max(x)
    return (x * 255).astype(np.uint8)


# discontinous volume will be splitted to continous chunks
def load_and_split_mri_from_dicom_dir(
        study_id,
        series_id,
        series_description,
):
    # 定义存放DICOM文件的目录路径
    dicom_dir = f'{kaggle_dir}/train_images/{study_id}/{series_id}'

    # 使用glob模块查找所有DICOM文件并进行自然排序
    dicom_file = natsorted(glob.glob(f'{dicom_dir}/*.dcm'))

    # 从文件名中提取实例编号
    instance_number = [int(f.split('/')[-1].split('.')[0]) for f in dicom_file]
    # 使用pydicom读取DICOM文件
    dicom = [pydicom.dcmread(f) for f in dicom_file]

    # 初始化用于存储DICOM数据的列表
    dicom_df = []

    # 遍历实例编号和DICOM文件元数据，创建包含所需信息的字典
    for i, d in zip(instance_number, dicom):
        try:
            dicom_df.append(
                dotdict(
                    study_id=study_id,
                    series_id=series_id,
                    series_description=series_description,
                    instance_number=i,
                    ImagePositionPatient=tuple([float(v) for v in d.ImagePositionPatient]),
                    ImageOrientationPatient=tuple([float(v) for v in d.ImageOrientationPatient]),
                    PixelSpacing=tuple([float(v) for v in d.PixelSpacing]),
                    SpacingBetweenSlices=float(d.SpacingBetweenSlices),
                    SliceThickness=float(d.SliceThickness),
                )
            )
        except AttributeError as e:
            print(f"Missing attribute in DICOM file {f}: {e}")
            continue

    if not dicom_df:
        raise ValueError(f"No valid DICOM files found in {dicom_dir}")

    # 将字典列表转换为pandas DataFrame
    dicom_df = pd.DataFrame(dicom_df)

    # 按ImageOrientationPatient字段分组并筛选，确保图像方向一致性
    if 'ImageOrientationPatient' not in dicom_df.columns:
        raise KeyError("ImageOrientationPatient column is missing in dicom_df")
    dicom_df = [d for _, d in dicom_df.groupby('ImageOrientationPatient')]

    # 初始化用于存储MRI数据的列表
    mri = []

    # 处理每个分组的DICOM数据
    for df in dicom_df:
        # 提取ImagePositionPatient和ImageOrientationPatient字段
        position = np.array(df['ImagePositionPatient'].values.tolist())
        orientation = np.array(df['ImageOrientationPatient'].values.tolist())

        # 计算法向量和投影位置
        normal = np.cross(orientation[:, :3], orientation[:, 3:])
        projection = np.sum(normal * position, 1)  # 计算投影

        # 将投影位置添加到DataFrame中，并按投影位置排序
        df.loc[:, 'projection'] = projection
        df = df.sort_values('projection')

        # 确保所有切片的参数是一致的
        assert len(df.SliceThickness.unique()) == 1
        assert len(df.ImageOrientationPatient.unique()) == 1
        assert len(df.SpacingBetweenSlices.unique()) == 1

        # 构建三维体积数据
        volume = [
            dicom[instance_number.index(i)].pixel_array for i in df.instance_number
        ]
        volume = np.stack(volume)

        # 归一化处理
        volume = normalise_to_8bit(volume)

        # 将处理后的DataFrame和体积数据添加到mri列表
        mri.append(dotdict(
            df=df,
            volume=volume,
        ))

    # 返回处理后的MRI数据列表
    return mri


# 这段代码的主要目的是为一个包含医学影像数据的 DataFrame 添加额外的信息，
# 包括图像的宽度（W）、高度（H）以及每个实例在三维空间中的位置（xx, yy, zz）。
# 这是通过读取相应的 DICOM 文件并使用其元数据来实现的。
# 计算空间坐标的公式考虑了图像的方向和像素间距，将二维图像坐标转换为三维世界坐标。
# convert 2d x,y to 3d X,Y,Z for point in label.csv
def add_XYZ_to_label_df(study_id_df):
    # 为DataFrame添加新的列，用于存放图像的宽度、高度以及世界坐标xx, yy, zz
    for col in ['W', 'H']:
        study_id_df.loc[:, col] = 0  # 初始化宽度和高度为0
    for col in ['xx', 'yy', 'zz']:
        study_id_df.loc[:, col] = 0.0  # 初始化世界坐标为0.0

    # 遍历DataFrame中的每一行数据
    for t, d in study_id_df.iterrows():
        # 构造DICOM文件的路径
        dicom_file = f'{kaggle_dir}/train_images/{d.study_id}/{d.series_id}/{d.instance_number}.dcm'
        # 使用pydicom读取DICOM文件
        dicom = pydicom.dcmread(dicom_file)

        # 从DICOM元数据中获取图像的宽度和高度
        H, W = dicom.pixel_array.shape
        # 获取DICOM文件中的患者图像位置
        sx, sy, sz = [float(v) for v in dicom.ImagePositionPatient]
        # 获取DICOM文件中的图像方向
        o0, o1, o2, o3, o4, o5 = [float(v) for v in dicom.ImageOrientationPatient]
        # 获取DICOM文件中的像素间距
        delx, dely = dicom.PixelSpacing

        # 根据图像方向和像素间距计算世界坐标xx, yy, zz
        xx = o0 * delx * d.x + o3 * dely * d.y + sx
        yy = o1 * delx * d.x + o4 * dely * d.y + sy
        zz = o2 * delx * d.x + o5 * dely * d.y + sz

        # 更新DataFrame中对应的列值
        study_id_df.loc[t, 'W'] = W
        study_id_df.loc[t, 'H'] = H
        study_id_df.loc[t, 'xx'] = xx
        study_id_df.loc[t, 'yy'] = yy
        study_id_df.loc[t, 'zz'] = zz

    # 返回更新后的DataFrame
    return study_id_df


# read all mri for one patient
def load_for_one(study_id_df):
    # 使用 groupby 对 DataFrame 进行分组，依据是 'series_description' 和 'series_id'
    # agg('first') 表示对每个分组应用 'first' 聚合函数，即选取每组的第一行数据
    # index 表示返回分组后各个组的唯一标识
    gb = study_id_df.groupby(['series_description', 'series_id']).agg('first').index

    # 初始化一个列表，用于存储每个序列的 MRI 数据
    mri = []

    # 遍历分组后得到的索引
    for series_description, series_id in gb:
        # 对于每个序列，调用 load_and_split_mri_from_dicom_dir 函数
        # 这个函数负责从 DICOM 目录加载数据，并将其分割成单独的 MRI 数据
        # 将结果添加到 mri 列表中
        mri += load_and_split_mri_from_dicom_dir(
            study_id=study_id,  # 假设 study_id 是一个全局变量或通过某种方式传入
            series_description=series_description,
            series_id=series_id
        )

    # 返回包含所有序列 MRI 数据的列表
    return mri


# back project 3D to 2d
# 这段代码的主要目的是将给定的世界坐标 (xx, yy, zz) 反投影（backproject）到 MRI 体积数据的二维坐标和实例编号上。
# 这是通过计算坐标的投影并检查它们是否在图像边界内来完成的。
# 如果坐标有效，函数返回 True 和对应的坐标以及实例编号；如果无效，返回 False 和默认值。
def backproject_XYZ(xx, yy, zz, mri):
    # 从传入的 mri 对象中获取相关信息
    r = mri
    # 获取分组后的第一行数据，包含序列的元数据
    d0 = r.df.iloc[0]

    # 提取 ImagePositionPatient 和 ImageOrientationPatient 字段
    sx, sy, sz = [float(v) for v in d0.ImagePositionPatient]
    o0, o1, o2, o3, o4, o5 = [float(v) for v in d0.ImageOrientationPatient]
    # 提取 PixelSpacing 和 SpacingBetweenSlices 字段
    delx, dely = d0.PixelSpacing
    delz = d0.SpacingBetweenSlices

    # 计算方向向量
    ax = np.array([o0, o1, o2])
    ay = np.array([o3, o4, o5])
    # 计算法向量 az 为 ax 和 ay 的叉乘结果
    az = np.cross(ax, ay)

    # 计算从世界坐标到图像坐标的向量 p
    p = np.array([xx - sx, yy - sy, zz - sz])
    # 将向量 p 投影到轴上，得到图像坐标 (x, y, z)
    x = np.dot(ax, p) / delx
    y = np.dot(ay, p) / dely
    z = np.dot(az, p) / delz
    # 将坐标四舍五入到最近的整数
    x = int(round(x))
    y = int(round(y))
    z = int(round(z))

    # 获取三维体积数据的维度
    D, H, W = r.volume.shape
    # 检查计算出的坐标是否在图像的边界内
    inside = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (z >= 0) & (z < D)
    if not inside:
        # 如果坐标超出边界，返回 False 和默认值
        return False, 0, 0, 0, 0

    # 如果坐标在边界内，返回 True 和计算出的坐标及对应的实例编号
    n = r.df.instance_number.values[z]
    return True, x, y, z, n


# load kaggle csv
desc_df = pd.read_csv(f'{kaggle_dir}/train_series_descriptions.csv')
label_df = pd.read_csv(f'{kaggle_dir}/train_label_coordinates.csv')
label_df = label_df.merge(desc_df, on=['study_id', 'series_id'])

# verify code

'''
study_id	series_id	series_description
0	4003253	702807833	Sagittal T2/STIR
1	4003253	1054713880	Sagittal T1
2	4003253	2448190387	Axial T2
'''

study_id = 4003253
study_id_df = label_df[label_df.study_id == study_id]
study_id_df = add_XYZ_to_label_df(study_id_df)
print(study_id_df.iloc[0])

mri = load_for_one(study_id_df)
print('len(mri): ', len(mri))
print('')

# backproject within the same view as truth point:
# - given x,y,z of view v from label.csv
# - project to 3d xx,yy,zz
# - for the same view v, backproject from xx,yy,zz to vx,vy,vz
# - check (vx,vy,vz) must be same as (x,y,z)

print('START VERIFICATION !!!')
for t, d in study_id_df.iterrows():
    print('=================================')
    # print(d)
    print('*****')
    xx, yy, zz = d.xx, d.yy, d.zz

    found = 0
    for r in mri:
        d0 = r.df.iloc[0]
        if not (
                (d0.study_id == d.study_id) &
                (d0.series_id == d.series_id) &
                (d.instance_number in r.df.instance_number)
        ): continue
        inside, x, y, z, n = backproject_XYZ(xx, yy, zz, r)
        found += 1
        print('truth:', d.instance_number, d.x, d.y)
        print('predict:', n, x, y, f'inside={inside}', f'array index z={z}')
    print(f'found={found}')
    print('')

    # example of correct cross view 3d to 2d projection
d = study_id_df.iloc[0]
print(d)
print('')

for r in mri:
    d0 = r.df.iloc[0]
    print(d0.series_id)
    print(d0.series_description)

    inside, x, y, z, n = backproject_XYZ(d.xx, d.yy, d.zz, r)
    print('predict:', n, x, y, f'inside={inside}', f'array index z={z}')

    if inside:
        slice = r.volume[z].copy()
        slice[y] = 255  # slice[y] // 2 + 127
        slice[:, x] = 255  # slice[:,x] // 2 + 127
    else:
        D, H, W = r.volume.shape
        slice = np.zeros((H, W), dtype=np.uint8)
    plt.imshow(slice, cmap='gray')
    plt.show()

# example of wrong cross view 3d to 2d projection ?????
study_id = 4096820034
study_id_df = label_df[label_df.study_id == study_id]
study_id_df = add_XYZ_to_label_df(study_id_df)
# print(study_id_df.iloc[0])

mri = load_for_one(study_id_df)
print('len(mri): ', len(mri))
print('')

d = study_id_df.iloc[0]
print(d)
print('')

for r in mri:
    d0 = r.df.iloc[0]
    print(d0.series_id)
    print(d0.series_description)

    inside, x, y, z, n = backproject_XYZ(d.xx, d.yy, d.zz, r)
    print('predict:', n, x, y, f'inside={inside}', f'array index z={z}')

    if inside:
        slice = r.volume[z].copy()
        slice[y] = 255  # slice[y] // 2 + 127
        slice[:, x] = 255  # slice[:,x] // 2 + 127
    else:
        D, H, W = r.volume.shape
        slice = np.zeros((H, W), dtype=np.uint8)
    plt.imshow(slice, cmap='gray')
    plt.show()

    # here is the magic !!!
# plot axial slices

color_table = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [0, 1, 1],
]

study_id = 40745534  # 88465004 #40745534 #4003253 #11340341 #4096820034
study_id_df = label_df[label_df.study_id == study_id]
study_id_df = add_XYZ_to_label_df(study_id_df)
study_id_df = study_id_df.sort_values(['series_id', 'instance_number'])
mri = load_for_one(study_id_df)
print('len(mri): ', len(mri))
print('')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_aspect('equal')

color_i = 0
for j, r in enumerate(mri):
    D, H, W = r.volume.shape
    if not 'Axial' in r.df.iloc[0].series_description: continue

    color = color_table[color_i]
    color_i += 1

    N = len(r.df)
    for i in range(N):
        d0 = r.df.iloc[i]

        o0, o1, o2, o3, o4, o5 = d0.ImageOrientationPatient
        ox = np.array([o0, o1, o2])
        oy = np.array([o3, o4, o5])
        sx, sy, sz = d0.ImagePositionPatient
        s = np.array([sx, sy, sz])
        delx, dely = d0.PixelSpacing

        p0 = s
        p1 = s + W * delx * ox
        p2 = s + H * dely * oy
        p3 = s + H * dely * oy + W * delx * ox

        grid = np.stack([p0, p1, p2, p3]).reshape(2, 2, 3)
        gx = grid[:, :, 0]
        gy = grid[:, :, 1]
        gz = grid[:, :, 2]

        if i == 0:
            ax.plot_surface(gx, gy, gz, alpha=0.7, color=color)
            ax.scatter([sx], [sy], [sz], color='black')

        else:
            ax.plot_surface(gx, gy, gz, alpha=0.3, color=color)
            ax.scatter([sx], [sy], [sz], alpha=0.1, color='black')

    # ---

    # plt.show()

xLabel = ax.set_xlabel('x')
yLabel = ax.set_ylabel('y')
zLabel = ax.set_zlabel('z')
plt.show()
zz = 0