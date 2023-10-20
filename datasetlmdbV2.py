from __future__ import print_function

import itertools
import os.path

import numpy as np
import lmdb
import cv2
import torch
import torch.utils.data as data
import tqdm
import warnings
warnings.filterwarnings("ignore")

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

class MITTrans():
    ###################################################################
    # Hor_Mean: (-0.03458570197876295, -0.005967754132065717)
    # Hor_Sed: (17486.81373507298, 17482.90288793422)

    # Ver_Mean: (0.00010218146492255243, 0.013796116097105104)
    # Hor_Sed: (11551.920305351046, 11550.725123982738)
    ###################################################################
    def __init__(self, hor_std=17484.0, ver_std=11551.0, ):
        self.hor_std = hor_std
        self.ver_std = ver_std

    def norm(self, arr):
        arr = (arr - np.mean(arr)) / np.std(arr)
        return arr

    def normalize(self, arr):
        arr_min, arr_max = np.min(arr, axis=0), np.max(arr, axis=0)
        arr = (arr - arr_min) / arr_max
        return arr

    def __call__(self, hor, ver, mask):
        # h, w = 154, 218
        if mask is not None:
            mask = cv2.resize(mask, (218, 154), None, None, interpolation=cv2.INTER_LINEAR)
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            mask = torch.from_numpy(mask)
            mask = mask.permute(2, 0, 1).float().contiguous()

        #hor, ver = torch.from_numpy(hor),  torch.from_numpy(ver)
        hor, ver = torch.from_numpy(hor/self.hor_std), torch.from_numpy(ver/self.ver_std)
        #print(hor.shape)
        hor = hor.permute(2, 3, 0, 1).float().contiguous()
        ver = ver.permute(2, 3, 0, 1).float().contiguous()
        return hor, ver, mask

class LMDBDataset(data.Dataset):
    def __init__(self, root, transform=None, mode='pretrain', keys=None,
                 start=None, end=None, views=[10], channels=12, task='mask'):
        self.root = [os.path.join(root, 'view{}'.format(str(i))) for i in views]
        self.mode = mode
        self.views = views

        ###################################################################
        #                               all 版本
        ###################################################################
        category_all = {
            'walk': {'1': [0, 0], '2': [1, 21], '3': [1, 13], '4': [1, 13], '5': [1, 13],      # 举例：'2': [1, 21]代表在view2中存在第1-21个帧序列是walk这个行为，'1': [0, 0]代表在view1中没有walk这个行为
                     '6': [0, 0], '7': [1, 13], '8': [0, 0], '9': [1, 13], '10': [0, 0]},
            'multi': {'1': [0, 0], '2': [65, 77], '3': [13, 22], '4': [13, 25], '5': [13, 25],
                      '6': [0, 0], '7': [13, 25], '8': [0, 0], '9': [13, 25], '10': [0, 0]},
            'styrofoam': {'1': [65, 68], '2': [37, 45], '3': [22, 30], '4': [25, 33], '5': [25, 33],
                          '6': [25, 33], '7': [25, 33], '8': [25, 33], '9': [25, 33], '10': [28, 36]},
            'carton': {'1': [62, 65], '2': [21, 29], '3': [30, 38], '4': [33, 41], '5': [33, 41],
                       '6': [33, 41], '7': [33, 41], '8': [33, 41], '9': [33, 41], '10': [36, 44]},
            'yoga': {'1': [0, 0], '2': [29, 37], '3': [39, 47], '4': [41, 49], '5': [41, 49],
                     '6': [41, 49], '7': [41, 49], '8': [41, 49], '9': [41, 49], '10': [44, 52]},
            'dark': {'1': [0, 0], '2': [0, 0], '3': [0, 0], '4': [0, 0], '5': [0, 0],
                     '6': [0, 0], '7': [0, 0], '8': [61, 71], '9': [61, 72], '10': [64, 74]},
            'action': {'1': [0, 0], '2': [45, 65], '3': [47, 62], '4': [49, 61], '5': [49, 61],
                       '6': [0, 0], '7': [49, 61], '8': [0, 0], '9': [49, 61], '10': [0, 0]}
        }


        ###################################################################
        #                               mask 版本
        ###################################################################
        category_mask = {
            'walk': {'1': [1, 36], '2': [1, 21], '3': [1, 13], '4': [1, 13], '5': [1, 13],
                     '6': [1, 13], '7': [1, 13], '8': [1, 13], '9': [1, 13], '10': [4, 16]},
            'multi': {'1': [77, 89], '2': [65, 77], '3': [13, 22], '4': [13, 25], '5': [13, 25],
                      '6': [13, 25], '7': [13, 25], '8': [13, 25], '9': [13, 25], '10': [16, 28]},
            'action': {'1': [37, 62], '2': [45, 65], '3': [47, 62], '4': [49, 61], '5': [49, 61],
                       '6': [49, 63], '7': [49, 61], '8': [49, 61], '9': [49, 61], '10': [52, 64]}
        }


        ###################################################################
        # Pose 版本(view1,6,8,10中没有keypoint数据被清洗)
        ###################################################################
        category_pose = {
            'walk': {'1': [1, 36], '2': [1, 21], '3': [1, 13], '4': [1, 13], '5': [1, 13],
                     '6': [1, 13], '7': [1, 13], '8': [1, 13], '9': [1, 13], '10': [4, 16]},
            'multi': {'1': [77, 89], '2': [65, 77], '3': [13, 22], '4': [13, 25], '5': [13, 25],
                      '6': [13, 25], '7': [13, 25], '8': [13, 25], '9': [13, 25], '10': [16, 28]},
            'styrofoam': {'1': [65, 68], '2': [37, 45], '3': [22, 30], '4': [25, 33], '5': [25, 33],
                          '6': [25, 29], '7': [25, 33], '8': [25, 28], '9': [25, 33], '10': [28, 32]},
            'carton': {'1': [62, 65], '2': [21, 29], '3': [30, 38], '4': [33, 41], '5': [33, 41],
                       '6': [33, 37], '7': [33, 41], '8': [33, 37], '9': [33, 41], '10': [36, 40]},
            'yoga': {'1': [0, 0], '2': [29, 37], '3': [39, 47], '4': [41, 49], '5': [41, 49],
                     '6': [41, 45], '7': [41, 49], '8': [41, 45], '9': [41, 49], '10': [44, 48]},
            'action': {'1': [37, 62], '2': [45, 65], '3': [47, 62], '4': [49, 61], '5': [49, 61],
                       '6': [49, 63], '7': [49, 61], '8': [49, 61], '9': [49, 61], '10': [52, 64]}
        }

        if task == 'mask':
            self.category = category_mask
            print('You are using the mask version of RF dataset')
        elif task == 'pose':
            self.category = category_pose
            print('You are using the pose version of RF dataset')
        else:
            self.category = category_all
            print('You are using the full version of RF dataset')


        if mode == 'pretrain':
            self.start = 0
            self.end = 590
        elif mode == 'train':
            self.start = 0
            self.end = 464
        else:
            self.start = 476
            self.end = 590
        self.start = start if not start is None else self.start
        self.end = end if not end is None else self.end
        self.task = task
        self.offset = 12  # 视频帧比雷达帧落后6（个视频帧）
        self.channels = channels
        self.keys = keys
        self.data_keys = self._get_keys()
        self.transform = transform
        self.env = []
        for path in self.root:
            self.env.append(lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False))

    def __len__(self):
        return len(self.data_keys)

    def _get_keys(self):
        keys = []
        for v in self.views:
            groups = []
            for k in self.keys:
                grp_clip = range(*self.category[k][str(v)])
                groups.extend(grp_clip)
            frames = range(self.start, self.end - self.channels, 2)
            d = itertools.product([v], groups, frames)
            d = list(d)
            keys.extend(d)
        keys = np.array(keys).reshape(-1, 3)
        return np.ascontiguousarray(keys)

    def load_radar(self, v, g, f):
        s, e = self.offset + f, self.channels + self.offset + f

        hkeys = ['h%02d_%04d_%04d' % (v, g, x) for x in range(s, e)]
        vkeys = ['v%02d_%04d_%04d' % (v, g, x) for x in range(s, e)]

        with self.env[self.views.index(v)].begin(write=False) as txn:
            radars = []
            for key in hkeys + vkeys:
                buf = txn.get(key.encode('ascii'))
                radar = np.frombuffer(buf, dtype=np.float32)
                radar = radar.reshape(160, 200, 2)
                radars.append(radar)
        hor = np.stack(radars[:len(radars) // 2], axis=-1)
        ver = np.stack(radars[len(radars) // 2:], axis=-1)
        hor = np.ascontiguousarray(hor)
        ver = np.ascontiguousarray(ver)
        return hor, ver

    def load_mask(self, v, g, f):
        s, e = f // 2, f // 2 + self.channels // 2
        mkeys = ['m%02d_%04d_%04d' % (v, g, x) for x in range(s, e)]

        with self.env[self.views.index(v)].begin(write=False) as txn:
            masks = []
            for key in mkeys:
                buf = txn.get(key.encode('ascii'))
                mask = np.frombuffer(buf, dtype=np.uint8)
                mask = mask.reshape(624, 820)
                masks.append(mask)
        mask = np.stack(masks, axis=-1)
        mask = np.ascontiguousarray(mask)
        return mask

    def double_boxes(self, boxes):
        new_boxes = np.zeros_like(boxes)
        w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
        h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
        w_double = 2 * w_half
        h_double = 2 * h_half
        x_c = (boxes[:, 2] + boxes[:, 0]) * .5
        y_c = (boxes[:, 3] + boxes[:, 1]) * .5
        new_boxes[:, 0] = x_c - w_double
        new_boxes[:, 1] = y_c - h_double
        new_boxes[:, 2] = x_c + w_double
        new_boxes[:, 3] = y_c + h_double
        return new_boxes


    def load_keypoint(self, v, g, f):
        kkey = 'k%02d_%04d_%04d' % (v, g, f // 2 + self.channels // 2)

        with self.env[self.views.index(v)].begin(write=False) as txn:
            buf = txn.get(kkey.encode('ascii'))
            if buf is None:
                print(kkey)
            trackid_offset_bbox_keypoints3d = np.frombuffer(buf, dtype=np.float32)
            #keypoint = keypoints_normal(keypoint)
            trackid_offset_bbox_keypoints3d = trackid_offset_bbox_keypoints3d.reshape(-1, 71)
            #trackid = trackid_offset_bbox_keypoints3d[:, 0]
            #offset  = trackid_offset_bbox_keypoints3d[:, 1:7]
            bbox = trackid_offset_bbox_keypoints3d[:, 7:15] #hx1,hy1,hx2,hy2, vx1,vy1,vx2,vy2
            hbbox = self.double_boxes(bbox[:, 0:4])         #box扩大为标注的2倍大小
            #vbbox = self.double_boxes(bbox[:, 4:])
            #hbbox = bbox[:, 0:4]

            keypoints = trackid_offset_bbox_keypoints3d[:, 15:]
            keypoints = keypoints.reshape(-1, 14, 4)[:, :, 0:3]  #修正后的keypoint的标注格式为[x, y, z, 1.0]
            target = {"view": int(v-1),
                      "boxes": torch.from_numpy(hbbox),
                      "labels": torch.ones(hbbox.shape[0], dtype=torch.int64),
                      "keypoints": torch.from_numpy(keypoints.reshape(-1, 42))
                      }
        return target

    def __getitem__(self, index):
        v, g, f = self.data_keys[index]
        hor, ver = self.load_radar(v, g, f)
        mask = None
        if self.mode == 'pretrain':
            if self.transform:
                hor, ver, mask = self.transform(hor, ver, mask)
            return [hor, ver], index
        else:
            if self.task == 'mask':
                lable = self.load_mask(v, g, f)
                if self.transform:
                    hor, ver, lable = self.transform(hor, ver, lable)
            elif self.task == 'pose':
                if self.transform:
                    hor, ver, mask = self.transform(hor, ver, mask)
                lable = self.load_keypoint(v, g, f)
            else:
                raise UserWarning('Task only includes RFPose or RFMask')
            return hor, ver, lable, index

    def resample(self, resample_ratio):
        self.stride = int(1.0 / resample_ratio)
        self.data_keys =self.data_keys[:: self.stride]


def pose_main(train_dataset):
    print(len(train_dataset))  # 2712个样本，每个样本是12帧的帧序列，序列中每个帧就是一个热图
    # print(train_dataset[100])

    inputs_ver, inputs_hor, targets, index = train_dataset[10]
    # print(type(inputs_hor), type(inputs_ver), type(targets), type(index))
    print(inputs_hor.shape, inputs_ver.shape, targets, index)
    # b = torch.randperm(12)[:2]
    # inputs_hor = torch.index_select(inputs_hor, 1, b)
    print(inputs_hor.shape)
    # print('labels:', targets['labels'])
    # print('keypoints:', targets['keypoints'].shape)
    # print('mask：', target.shape, targets[0])   # 在mask下target的shape是[6,160,224]，不是雷达帧序列的长度12，因为雷达的两帧对应图像的一帧，所以除以2等于6，后两位就代表分割图
    ver = inputs_ver[:, 0, :, :]  # 一个帧序列共12帧，这里取出最后一帧
    hor = inputs_hor[:, 0, :, :]

    boxes = targets['boxes']

    ver_abs = np.sqrt(ver[0]**2 + ver[1]**2)      # 直接把复数值的虚部与实部给均方
    hor_abs = np.sqrt(hor[0] ** 2 + hor[1] ** 2)
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)   # f为matplotlib.figure.Figure 对象。ax：子图对象（ matplotlib.axes.Axes）或者是他的数组

    print("ver_abs:", ver_abs.shape)
    ax1.imshow(ver_abs)
    w = (boxes[:, 2] - boxes[:, 0])
    h = (boxes[:, 3] - boxes[:, 1])
    x = (boxes[:, 2] + boxes[:, 0]) * .5 - 0.5*w
    y = (boxes[:, 3] + boxes[:, 1]) * .5 - 0.5*h
    ax1.add_patch(      # 加一个边界框
        patches.Rectangle(
            (x, y),  # (x,y)
            w,  # width
            h,  # height
            linewidth=2, edgecolor='r', facecolor='none'
        )
    )

    r_off, z_min, z_max = 112, 85.96260799, 128.65236639
    v = boxes
    for i in [1, 3]:
        y = (160 - boxes[:, i]) ** 2 + (boxes[:, 0] / 2 + boxes[:, 2] / 2 - r_off) ** 2
        v[:, i] = 160 - y ** 0.5
    v[:, 0] = z_min
    v[:, 2] = z_max

    w = (v[:, 2] - v[:, 0])
    h = (v[:, 3] - v[:, 1])
    x = (v[:, 2] + v[:, 0]) * .5 - 0.5*w
    y = (v[:, 3] + v[:, 1]) * .5 - 0.5*h

    ax2.imshow(hor_abs)
    ax2.add_patch(
        patches.Rectangle(
            (x, y),  # (x,y)
            w,  # width
            h,  # height
            linewidth=2, edgecolor='r', facecolor='none'
        )
    )
    plt.show()

def mask_main(train_dataset):
    print(len(train_dataset))  # 2712个样本，每个样本是12帧的帧序列，序列中每个帧就是一个热图
    # print(train_dataset[100])

    inputs_hor, inputs_ver, inputs_mask, index = train_dataset[50]
    # print(type(inputs_hor), type(inputs_ver), type(targets), type(index))
    print(inputs_hor.shape, inputs_ver.shape, inputs_mask.shape, index)
    # b = torch.randperm(12)[:2]
    # inputs_hor = torch.index_select(inputs_hor, 1, b)
    print(inputs_hor.shape)
    # print('labels:', targets['labels'])
    # print('mask：', target.shape, targets[0])   # 在mask下target的shape是[6,160,224]，不是雷达帧序列的长度12，因为雷达的两帧对应图像的一帧，所以除以2等于6，后两位就代表分割图
    ver = inputs_hor[:, 11, :, :]  # 一个帧序列共12帧，这里取出最后一帧
    hor = inputs_ver[:, 11, :, :]

    ver_abs = np.sqrt(ver[0]**2 + ver[1]**2)      # 直接把复数值的虚部与实部给均方
    hor_abs = np.sqrt(hor[0] ** 2 + hor[1] ** 2)
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)   # f为matplotlib.figure.Figure 对象。ax：子图对象（ matplotlib.axes.Axes）或者是他的数组

    print("ver_abs:", ver_abs.shape)
    ax1.imshow(ver_abs)
    ax2.imshow(hor_abs)
    plt.show()

def mask_mutilfusion_main(train_dataset):
    print(len(train_dataset))  # 2712个样本，每个样本是12帧的帧序列，序列中每个帧就是一个热图

    inputs_ver, inputs_hor, inputs_mask, index = train_dataset[60]
    print(inputs_hor.shape, inputs_ver.shape, inputs_mask.shape, index)

    # 从50帧中随机选取几帧
    prob = np.array([0.15, 0.85])
    index = np.random.choice([1, 2], p=prob.ravel())
    print('index:', index)
    a = 1
    if int(index) == 2:
        a = np.random.randint(2, 50)

    b = torch.randperm(50)[:a]
    print(a, b, b.shape)

    inputs_hor = torch.index_select(inputs_hor, 1, b)
    inputs_ver = torch.index_select(inputs_ver, 1, b)
    print("inputs_hor：", inputs_hor.shape)
    print("inputs_ver：", inputs_ver.shape)

    # 先求和
    inputs_hor = np.sqrt(inputs_hor[0] ** 2 + inputs_hor[1] ** 2)
    inputs_ver = np.sqrt(inputs_ver[0] ** 2 + inputs_ver[1] ** 2)
    print("inputs_ver1：", inputs_ver.shape)
    print("inputs_hor1：", inputs_hor.shape)

    inputs_hor = torch.sum(inputs_hor, dim=0)
    inputs_ver = torch.sum(inputs_ver, dim=0)
    print("inputs_hor2：", inputs_hor.shape)
    print("inputs_ver2：", inputs_ver.shape)

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)   # f为matplotlib.figure.Figure 对象。ax：子图对象（ matplotlib.axes.Axes）或者是他的数组

    ax1.imshow(inputs_ver)
    ax2.imshow(inputs_hor)
    plt.show()


if __name__=='__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import ArtistAnimation  # 导入负责绘制动画的接口
    import random

    seed_torch()

    train_dataset = LMDBDataset(r'E:\RF',
                                transform=MITTrans(),
                                views=[10],
                                channels=50,
                                mode='train',
                                task='pose',       # 切换mask和pose，会导致输出不同
                                keys=['walk', 'multi', 'action'])
    pose_main(train_dataset)
    # mask_main(train_dataset)
    # mask_mutilfusion_main(train_dataset)