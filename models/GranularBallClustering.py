import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


class GB:
    def __init__(self, points, label):
        self.points = points
        self.data = np.array([p.data for p in points])
        self.center = self.data.mean(0)
        self.radius = self.get_radius()
        self.flag = 0
        self.label = label
        self.num = len(self.data)
        self.out = 0
        self.size = 1
        self.overlap = 0
        self.hardlapcount = 0
        self.softlapcount = 0

    def get_radius(self):
        return max(((self.data - self.center) ** 2).sum(axis=1) ** 0.5)


class UF:
    def __init__(self, len):
        self.parent = [0] * len
        self.size = [0] * len
        self.count = len

        for i in range(0, len):
            self.parent[i] = i
            self.size[i] = 1

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP == rootQ:
            return
        if self.size[rootP] > self.size[rootQ]:
            self.parent[rootQ] = rootP
            self.size[rootP] += self.size[rootQ]
        else:
            self.parent[rootP] = rootQ
            self.size[rootQ] += self.size[rootP]
        self.count = self.count - 1

    def connected(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        return rootP == rootQ

    def count(self):
        return self.count


def division(hb_list):
    gb_list_new = []
    for hb in hb_list:
        if len(hb) >= 8:
            ball_1, ball_2 = spilt_ball(hb)
            DM_parent = get_DM(hb)
            DM_child_1 = get_DM(ball_1)
            DM_child_2 = get_DM(ball_2)
            t1 = ((DM_child_1 > DM_parent) & (DM_child_2 > DM_parent))
            if t1:
                gb_list_new.extend([ball_1, ball_2])
            else:
                gb_list_new.append(hb)
        else:
            gb_list_new.append(hb)

    return gb_list_new


# 参数是每个gb, DataPoint对象列表
def spilt_ball_2(gb):
    ball1 = []
    ball2 = []
    data = np.array([p.data for p in [points for points in gb]])
    n, m = data.shape
    X = data.T
    G = np.dot(X.T, X)
    H = np.tile(np.diag(G), (n, 1))
    D = np.sqrt(np.abs(H + H.T - G * 2))
    r, c = np.where(D == np.max(D))
    r1 = r[1]
    c1 = c[1]
    for j in range(0, len(data)):
        if D[j, r1] < D[j, c1]:
            ball1.append(gb[j])
        else:
            ball2.append(gb[j])
    ball1 = np.array(ball1)
    ball2 = np.array(ball2)
    return [ball1, ball2]


def get_density_volume(gb):
    num = len(gb)
    data = np.array([p.data for p in [points for points in gb]])
    center = data.mean(0)
    diffMat = np.tile(center, (num, 1)) - data
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sum_radius = 0
    if len(distances) == 0:
        print("0")
    radius = max(distances)
    for i in distances:
        sum_radius = sum_radius + i
    mean_radius = sum_radius / num
    dimension = len(data[0])
    # print('*******dimension********',dimension)
    if mean_radius != 0:
        density_volume = num / sum_radius
    else:
        density_volume = num

    return density_volume


# 无参遍历粒球是否需要分裂，根据子球和父球的比较，不带断裂判断的分裂,1分2
def division_2_2(gb_list, n):
    gb_list_new_2 = []

    for gb in gb_list:
        if len(gb) >= 8:
            ball_1, ball_2 = spilt_ball_2(gb)
            if len(ball_1) * len(ball_2) == 0:
                return gb_list
            density_parent = get_density_volume(gb)
            density_child_1 = get_density_volume(ball_1)
            density_child_2 = get_density_volume(ball_2)
            w = len(ball_1) + len(ball_2)
            w1 = len(ball_1) / w
            w2 = len(ball_2) / w
            w_child = (w1 * density_child_1 + w2 * density_child_2)
            t1 = ((density_child_1 > density_parent) & (density_child_2 > density_parent))
            t2 = (w_child > density_parent)
            t3 = ((len(ball_1) > 0) & (len(ball_2) > 0))  # 球中数据个数低于4个的情况不能分裂
            if t2:
                gb_list_new_2.extend([ball_1, ball_2])
            else:
                gb_list_new_2.append(gb)
        else:
            gb_list_new_2.append(gb)

    return gb_list_new_2


# 粒球分裂
def spilt_ball(gb):
    ball1 = []
    ball2 = []
    data = np.array([p.data for p in [points for points in gb]])
    n, m = data.shape
    X = data.T
    G = np.dot(X.T, X)
    H = np.tile(np.diag(G), (n, 1))
    D = np.sqrt(np.abs(H + H.T - G * 2))
    r, c = np.where(D == np.max(D))
    r1 = r[1]
    c1 = c[1]
    for j in range(0, len(data)):
        if D[j, r1] < D[j, c1]:
            ball1.append(gb[j])
        else:
            ball2.append(gb[j])
    ball1 = np.array(ball1)
    ball2 = np.array(ball2)
    return [ball1, ball2]


def get_DM(hb):
    num = len(hb)
    center = hb.mean(0)
    diffMat = np.tile(center, (num, 1)) - hb
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sum_radius = 0
    radius = max(distances)
    for i in distances:
        sum_radius = sum_radius + i
    mean_radius = sum_radius / num
    dimension = len(hb[0])
    if mean_radius != 0:
        DM = num / sum_radius
    else:
        DM = num
    return DM


def get_radius(hb):
    num = len(hb)
    center = hb.mean(0)
    diffMat = np.tile(center, (num, 1)) - hb
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    radius = max(distances)
    return radius


def plot_dot(data):
    fig = plt.subplot(121)
    try:
        fig.scatter(data[:, 0], data[:, 1], s=7, c="#314300", linewidths=5, alpha=0.6, marker='o', label='data point')
    except:
        print()
    fig.legend()
    return fig.findobj()


def hb_plot(gbs, noise):
    color = {
        0: '#707afa',
        1: '#ffe135',
        2: '#16ccd0',
        3: '#ed7231',
        4: '#0081cf',
        5: '#afbed1',
        6: '#bc0227',
        7: '#d4e7bd',
        8: '#f8d7aa',
        9: '#fecf45',
        10: '#f1f1b8',
        11: '#b8f1ed',
        12: '#ef5767',
        13: '#e7bdca',
        14: '#8e7dfa',
        15: '#d9d9fc',
        16: '#2cfa41',
        17: '#e96d29',
        18: '#7f722f',
        19: '#bd57fa',
        20: '#e4f788',
        21: '#fb8e94',
        22: '#b8d38f',
        23: '#e3a04f',
        24: '#edc02f',
        25: '#ff8444', }
    label_c = {
        0: 'cluster-1',
        1: 'cluster-2',
        2: 'cluster-3',
        3: 'cluster-4',
        4: 'cluster-5',
        5: 'cluster-6',
        6: 'cluster-7',
        7: 'cluster-8',
        8: 'cluster-9',
        9: 'cluster-10',
        10: 'cluster-11',
        11: 'cluster-12',
        12: 'cluster-13',
        13: 'cluster-14',
        14: 'cluster-15',
        15: 'cluster-16',
        16: 'cluster-17',
        17: 'cluster-18',
        18: 'cluster-19',
        19: 'cluster-20',
        20: 'cluster-21',
        21: 'cluster-22',
        22: 'cluster-23',
        23: 'cluster-24',
        24: 'cluster-25'}
    plt.figure(figsize=(10, 10))
    label_num = {}
    for i in range(0, len(gbs)):
        label_num.setdefault(gbs[i].label, 0)
        label_num[gbs[i].label] = label_num.get(gbs[i].label) + len(gbs[i].data)

    label = set()
    for key in label_num.keys():
        label.add(key)
    list = []
    for i in range(0, len(label)):
        list.append(label.pop())

    for i in range(0, len(list)):
        if list[i] == -1:
            list.remove(-1)
            break

    for i in range(0, len(list)):
        if i >= 25:
            c = "black"
            label = "too many cluster"
        else:
            c = color[i]
            label = label_c[i]
        for key in gbs.keys():
            if gbs[key].label == list[i]:
                plt.scatter(gbs[key].data[:, 0], gbs[key].data[:, 1], s=4, c=color[i], linewidths=5, alpha=0.9,
                            marker='o', label=label_c[i])
                break

    for key in gbs.keys():
        for i in range(0, len(list)):
            if i >= 25:
                c = "black"
                label = "too many cluster"
            else:
                c = color[i]
                label = label_c[i]
            if gbs[key].label == list[i]:
                plt.scatter(gbs[key].data[:, 0], gbs[key].data[:, 1], s=4, c=color[i], linewidths=5, alpha=0.9,
                            marker='o')
    if len(noise) > 0:
        plt.scatter(noise[:, 0], noise[:, 1], s=40, c='black', linewidths=2, alpha=1, marker='x', label='noise')

    for key in gbs.keys():
        for i in range(0, len(list)):
            if gbs[key].label == -1:
                plt.scatter(gbs[key].data[:, 0], gbs[key].data[:, 1], s=40, c='black', linewidths=2, alpha=1,
                            marker='x')
    plt.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False
    )
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
    plt.show()


def draw_ball(hb_list):
    fig = plt.subplot(121)
    for data in hb_list:
        if len(data) > 1:
            center = data.mean(0)
            radius = np.max((((data - center) ** 2).sum(axis=1) ** 0.5))
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            fig.plot(x, y, ls='-', color='black', lw=0.7)
        else:
            fig.plot(data[0][0], data[0][1], marker='*', color='#0000EF', markersize=3)
    fig.legend(loc=1)
    return fig.findobj()


def connect_ball0(hb_list, noise, c_count):
    hb_cluster = {}
    for i in range(0, len(hb_list)):
        hb = GB(hb_list[i], i)
        hb_cluster[i] = hb

    radius_sum = 0  # 总半径
    num_sum = 0  # 总样本数
    hb_len = 0  # 总粒球数
    radius_sum = 0
    num_sum = 0
    for i in range(0, len(hb_cluster)):
        if hb_cluster[i].out == 0:
            hb_len = hb_len + 1
            radius_sum = radius_sum + hb_cluster[i].radius
            num_sum = num_sum + hb_cluster[i].num

    for i in range(0, len(hb_cluster) - 1):
        if hb_cluster[i].out != 1:
            center_i = hb_cluster[i].center
            radius_i = hb_cluster[i].radius
            for j in range(i + 1, len(hb_cluster)):
                if hb_cluster[j].out != 1:
                    center_j = hb_cluster[j].center
                    radius_j = hb_cluster[j].radius
                    dis = ((center_i - center_j) ** 2).sum(axis=0) ** 0.5
                    if (dis <= radius_i + radius_j) & ((hb_cluster[i].hardlapcount == 0) & (
                            hb_cluster[j].hardlapcount == 0)):
                        hb_cluster[i].overlap = 1
                        hb_cluster[j].overlap = 1
                        hb_cluster[i].hardlapcount = hb_cluster[i].hardlapcount + 1
                        hb_cluster[j].hardlapcount = hb_cluster[j].hardlapcount + 1

    hb_uf = UF(len(hb_list))
    for i in range(0, len(hb_cluster) - 1):
        if hb_cluster[i].out != 1:
            center_i = hb_cluster[i].center
            radius_i = hb_cluster[i].radius
            for j in range(i + 1, len(hb_cluster)):
                if hb_cluster[j].out != 1:
                    center_j = hb_cluster[j].center
                    radius_j = hb_cluster[j].radius
                    max_radius = max(radius_i, radius_j)
                    min_radius = min(radius_i, radius_j)
                    dis = ((center_i - center_j) ** 2).sum(axis=0) ** 0.5
                    if c_count == 1:
                        dynamic_overlap = dis < radius_i + radius_j + 1 * min_radius / (
                                min(hb_cluster[i].hardlapcount, hb_cluster[j].hardlapcount) + 1)
                    if c_count == 2:
                        dynamic_overlap = dis <= radius_i + radius_j + 1 * max_radius / (
                                min(hb_cluster[i].hardlapcount, hb_cluster[j].hardlapcount) + 1)
                    num_limit = ((hb_cluster[i].num > 2) & (hb_cluster[j].num > 2))
                    if dynamic_overlap & num_limit:
                        hb_cluster[i].flag = 1
                        hb_cluster[j].flag = 1
                        hb_uf.union(i, j)
                    if dis <= radius_i + radius_j + max_radius:
                        hb_cluster[i].softlapcount = 1
                        hb_cluster[j].softlapcount = 1

    for i in range(0, len(hb_cluster)):
        k = i
        if hb_uf.parent[i] != i:
            while hb_uf.parent[k] != k:
                k = hb_uf.parent[k]
        hb_uf.parent[i] = k

    for i in range(0, len(hb_cluster)):
        hb_cluster[i].label = hb_uf.parent[i]
        hb_cluster[i].size = hb_uf.size[i]

    label_num = set()
    for i in range(0, len(hb_cluster)):
        label_num.add(hb_cluster[i].label)

    list = []
    for i in range(0, len(label_num)):
        list.append(label_num.pop())

    for i in range(0, len(hb_cluster)):
        if (hb_cluster[i].hardlapcount == 0) & (hb_cluster[i].softlapcount == 0):
            hb_cluster[i].flag = 0

    for i in range(0, len(list)):
        count_ball = 0
        count_data = 0
        list1 = []
        for key in range(0, len(hb_cluster)):
            if hb_cluster[key].label == list[i]:
                count_ball += 1
                count_data += hb_cluster[key].num
                list1.append(key)
        while count_ball < 6:
            for j in range(0, len(list1)):
                hb_cluster[list1[j]].flag = 0
            break

    for i in range(0, len(hb_cluster)):
        distance = np.sqrt(2)
        if hb_cluster[i].flag == 0:
            for j in range(0, len(hb_cluster)):
                if hb_cluster[j].flag == 1:
                    center = hb_cluster[i].center
                    center2 = hb_cluster[j].center
                    dis = ((center - center2) ** 2).sum(axis=0) ** 0.5 - (hb_cluster[i].radius + hb_cluster[j].radius)
                    if dis < distance:
                        distance = dis
                        hb_cluster[i].label = hb_cluster[j].label
                        hb_cluster[i].flag = 2
            for k in range(0, len(noise)):
                center = hb_cluster[i].center
                dis = ((center - noise[k]) ** 2).sum(axis=0) ** 0.5
                if dis < distance:
                    distance = dis
                    hb_cluster[i].label = -1
                    hb_cluster[i].flag = 2

    label_num = set()
    for i in range(0, len(hb_cluster)):
        label_num.add(hb_cluster[i].label)
    return hb_cluster


def connect_ball0(gb_list, noise, c_count):
    gb_dist = {}
    for i in range(0, len(gb_list)):
        gb = GB(gb_list[i], i)
        gb_dist[i] = gb

    radius_sum = 0
    num_sum = 0

    gblen = 0
    radius_sum = 0
    num_sum = 0
    for i in range(0, len(gb_dist)):
        if gb_dist[i].out == 0:
            gblen = gblen + 1
            radius_sum = radius_sum + gb_dist[i].radius
            num_sum = num_sum + gb_dist[i].num

    for i in range(0, len(gb_dist) - 1):
        if gb_dist[i].out != 1:
            center_i = gb_dist[i].center
            radius_i = gb_dist[i].radius
            for j in range(i + 1, len(gb_dist)):
                if gb_dist[j].out != 1:
                    center_j = gb_dist[j].center
                    radius_j = gb_dist[j].radius
                    dis = ((center_i - center_j) ** 2).sum(axis=0) ** 0.5
                    if (dis <= radius_i + radius_j) & (
                            (gb_dist[i].hardlapcount == 0) & (gb_dist[j].hardlapcount == 0)):
                        gb_dist[i].overlap = 1
                        gb_dist[j].overlap = 1
                        gb_dist[i].hardlapcount = gb_dist[i].hardlapcount + 1
                        gb_dist[j].hardlapcount = gb_dist[j].hardlapcount + 1
    gb_uf = UF(len(gb_list))
    for i in range(0, len(gb_dist) - 1):
        if gb_dist[i].out != 1:
            center_i = gb_dist[i].center
            radius_i = gb_dist[i].radius
            for j in range(i + 1, len(gb_dist)):
                if gb_dist[j].out != 1:
                    center_j = gb_dist[j].center
                    radius_j = gb_dist[j].radius
                    max_radius = max(radius_i, radius_j)
                    min_radius = min(radius_i, radius_j)
                    dis = ((center_i - center_j) ** 2).sum(axis=0) ** 0.5
                    if c_count == 1:
                        dynamic_overlap = dis < radius_i + radius_j + 1 * min_radius / (
                                max(gb_dist[i].hardlapcount, gb_dist[j].hardlapcount) + 1)
                    if c_count == 2:
                        dynamic_overlap = dis <= radius_i + radius_j + 1 * min_radius / (
                                max(gb_dist[i].hardlapcount, gb_dist[j].hardlapcount) + 1)
                    num_limit = ((gb_dist[i].num > 2) & (gb_dist[j].num > 2))
                    if dynamic_overlap:
                        gb_dist[i].flag = 1
                        gb_dist[j].flag = 1
                        gb_uf.union(i, j)
                    if dis <= radius_i + radius_j + 3 * max_radius:
                        gb_dist[i].softlapcount += 1
                        gb_dist[j].softlapcount += 1
                        gb_uf.union(i, j)

    for i in range(0, len(gb_dist)):
        k = i
        if gb_uf.parent[i] != i:
            while gb_uf.parent[k] != k:
                k = gb_uf.parent[k]
        gb_uf.parent[i] = k

    for i in range(0, len(gb_dist)):
        gb_dist[i].label = gb_uf.parent[i]
        gb_dist[i].size = gb_uf.size[i]

    label_num = set()
    for i in range(0, len(gb_dist)):
        label_num.add(gb_dist[i].label)

    list = []
    for i in range(0, len(label_num)):
        list.append(label_num.pop())

    for i in range(0, len(gb_dist)):
        if (gb_dist[i].hardlapcount == 0) and (gb_dist[i].softlapcount == 0):
            gb_dist[i].flag = 0
    for i in range(0, len(gb_dist)):
        distance = np.sqrt(2)
        if gb_dist[i].flag == 0:
            for j in range(0, len(gb_dist)):
                if gb_dist[j].flag == 1:
                    center = gb_dist[i].center
                    center2 = gb_dist[j].center
                    dis = ((center - center2) ** 2).sum(axis=0) ** 0.5 - (gb_dist[i].radius + gb_dist[j].radius)
                    if dis < distance:
                        distance = dis
                        gb_dist[i].label = gb_dist[j].label
                        gb_dist[i].flag = 2
            for k in range(0, len(noise)):
                center = gb_dist[i].center
                dis = ((center - noise[k]) ** 2).sum(axis=0) ** 0.5
                if dis < distance:
                    distance = dis
                    gb_dist[i].label = -1
                    gb_dist[i].flag = 2

    label_num = set()
    for i in range(0, len(gb_dist)):
        label_num.add(gb_dist[i].label)
    return gb_dist


def connect_ball(gb_list, noise, c_count):
    gb_dist = {}
    for i in range(0, len(gb_list)):
        points = np.array([point for point in gb_list[i]])
        gb = GB(points, i)
        gb_dist[i] = gb

    radius_sum = 0
    num_sum = 0

    gblen = 0
    radius_sum = 0
    num_sum = 0
    for i in range(0, len(gb_dist)):
        if gb_dist[i].out == 0:
            gblen = gblen + 1
            radius_sum = radius_sum + gb_dist[i].radius
            num_sum = num_sum + gb_dist[i].num

    for i in range(0, len(gb_dist) - 1):
        if gb_dist[i].out != 1:
            center_i = gb_dist[i].center
            radius_i = gb_dist[i].radius
            for j in range(i + 1, len(gb_dist)):
                if gb_dist[j].out != 1:
                    center_j = gb_dist[j].center
                    radius_j = gb_dist[j].radius
                    dis = ((center_i - center_j) ** 2).sum(axis=0) ** 0.5
                    if (dis <= radius_i + radius_j) & (
                            (gb_dist[i].hardlapcount == 0) & (gb_dist[j].hardlapcount == 0)):  # 由于两个的点是噪声球，所以纳入重叠统计
                        gb_dist[i].overlap = 1
                        gb_dist[j].overlap = 1
                        gb_dist[i].hardlapcount = gb_dist[i].hardlapcount + 1
                        gb_dist[j].hardlapcount = gb_dist[j].hardlapcount + 1
    gb_uf = UF(len(gb_list))
    for i in range(0, len(gb_dist) - 1):
        if gb_dist[i].out != 1:
            center_i = gb_dist[i].center
            radius_i = gb_dist[i].radius
            for j in range(i + 1, len(gb_dist)):
                if gb_dist[j].out != 1:
                    center_j = gb_dist[j].center
                    radius_j = gb_dist[j].radius
                    max_radius = max(radius_i, radius_j)
                    min_radius = min(radius_i, radius_j)
                    dis = ((center_i - center_j) ** 2).sum(axis=0) ** 0.5
                    if c_count == 1:
                        dynamic_overlap = dis < radius_i + radius_j + 1 * min_radius / (
                                min(gb_dist[i].hardlapcount, gb_dist[j].hardlapcount) + 1)
                    if c_count == 2:
                        dynamic_overlap = dis <= radius_i + radius_j + 1 * min_radius / (
                                min(gb_dist[i].hardlapcount, gb_dist[j].hardlapcount) + 1)
                    num_limit = ((gb_dist[i].num > 2) & (gb_dist[j].num > 2))
                    if dynamic_overlap:
                        gb_dist[i].flag = 1
                        gb_dist[j].flag = 1
                        gb_uf.union(i, j)
                    if dis <= radius_i + radius_j + 3 * max_radius:
                        gb_dist[i].softlapcount += 1
                        gb_dist[j].softlapcount += 1
                        gb_uf.union(i, j)

    for i in range(0, len(gb_dist)):
        k = i
        if gb_uf.parent[i] != i:
            while (gb_uf.parent[k] != k):
                k = gb_uf.parent[k]
        gb_uf.parent[i] = k

    for i in range(0, len(gb_dist)):
        gb_dist[i].label = gb_uf.parent[i]
        gb_dist[i].size = gb_uf.size[i]

    label_num = set()
    for i in range(0, len(gb_dist)):
        label_num.add(gb_dist[i].label)

    list = []
    for i in range(0, len(label_num)):
        list.append(label_num.pop())

    for i in range(0, len(gb_dist)):
        distance = np.sqrt(2)
        if gb_dist[i].flag == 0:
            for j in range(0, len(gb_dist)):
                if gb_dist[j].flag == 1:
                    center = gb_dist[i].center
                    center2 = gb_dist[j].center
                    dis = ((center - center2) ** 2).sum(axis=0) ** 0.5 - (gb_dist[i].radius + gb_dist[j].radius)
                    if dis < distance:
                        distance = dis
                        gb_dist[i].label = gb_dist[j].label
                        gb_dist[i].flag = 2
            for k in range(0, len(noise)):
                center = gb_dist[i].center
                dis = ((center - noise[k]) ** 2).sum(axis=0) ** 0.5
                if dis < distance:
                    distance = dis
                    gb_dist[i].label = -1
                    gb_dist[i].flag = 2

    label_num = set()
    for i in range(0, len(gb_dist)):
        label_num.add(gb_dist[i].label)
    return gb_dist


def normalized_ball(hb_list, radius_detect):
    hb_list_temp = []
    for hb in hb_list:
        if len(hb) < 2:
            hb_list_temp.append(np.array(hb))
        else:
            if get_radius(np.array([p.data for p in hb])) <= 1.5 * radius_detect:
                hb_list_temp.append(np.array(hb))
            else:
                ball_1, ball_2 = spilt_ball(hb)
                hb_list_temp.extend([ball_1, ball_2])
    return hb_list_temp


def normalized_ball_2(hb_cluster, radius_mean, list1):
    hb_list_temp = []
    for i in range(0, len(radius_mean)):
        for key in hb_cluster.keys():
            if hb_cluster[key].label == list1[i]:
                if hb_cluster[key].num < 2:
                    hb_list_temp.append(hb_cluster[key].points)
                else:
                    ball_1, ball_2 = spilt_ball(hb_cluster[key].points)
                    if hb_cluster[key].radius <= 1.5 * radius_mean[i] or len(ball_1) * len(ball_2) == 0:
                        hb_list_temp.append(hb_cluster[key].points)
                    else:
                        hb_list_temp.extend([ball_1, ball_2])
    return hb_list_temp


def load_data(key):
    dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(dir + "/synthetic/" + key + ".csv", header=None)
    data = df.values
    return data


def main(data):
    hb_list_temp = [data]
    row = np.shape(hb_list_temp)[0]
    col = np.shape(hb_list_temp)[1]
    n = row * col

    while 1:
        ball_number_old = len(hb_list_temp)
        hb_list_temp = division_2_2(hb_list_temp, n)
        ball_number_new = len(hb_list_temp)
        if ball_number_new == ball_number_old:
            break

    radius = []
    for hb in hb_list_temp:
        if len(hb) >= 2:
            radius.append(get_radius(np.array([p.data for p in hb])))

    radius_median = np.median(radius)
    radius_mean = np.mean(radius)
    radius_detect = max(radius_median, radius_mean)

    while 1:
        ball_number_old = len(hb_list_temp)
        hb_list_temp = normalized_ball(hb_list_temp, radius_detect)
        ball_number_new = len(hb_list_temp)
        if ball_number_new == ball_number_old:
            break
    noise = []
    hb_cluster = connect_ball(hb_list_temp, noise, 1)

    label_num = {}
    for i in range(0, len(hb_cluster)):
        label_num.setdefault(hb_cluster[i].label, 0)
        label_num[hb_cluster[i].label] = label_num.get(hb_cluster[i].label) + len(hb_cluster[i].data)

    label = set()
    for key in label_num.keys():
        if label_num[key] > 2:
            label.add(key)
    list1 = []
    for i in range(0, len(label)):
        list1.append(label.pop())
    radius_detect = [0] * len(list1)
    count_cluster_num = [0] * len(list1)
    radius_mean = [0] * len(list1)
    for key in hb_cluster.keys():
        for i in range(0, len(list1)):
            if hb_cluster[key].label == list1[i]:
                radius_detect[i] = radius_detect[i] + hb_cluster[key].radius
                count_cluster_num[i] = count_cluster_num[i] + 1

    for i in range(0, len(list1)):
        radius_mean[i] = radius_detect[i] / count_cluster_num[i]

    while 1:
        ball_number_old = len(hb_list_temp)
        hb_list_temp = normalized_ball_2(hb_cluster, radius_mean, list1)
        ball_number_new = len(hb_list_temp)
        if ball_number_new == ball_number_old:
            break
    gb_list_final = hb_list_temp
    noise = []

    gb_list_cluster = connect_ball(gb_list_final, noise, 2)
    hb_plot(gb_list_cluster, noise)
    return gb_list_final, gb_list_cluster
