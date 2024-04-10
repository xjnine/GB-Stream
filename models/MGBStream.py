import csv
import sys
from itertools import groupby
import psutil
import time
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from models.GranularBallClustering import *
from models.MGB import *


class MGBStream:
    def __init__(self, raw_data, name, threshold, velocity=1000):
        self.datasetName = name
        self.threshold = threshold
        self.velocity = velocity
        self.data = self.normalized(raw_data)
        self.image_list = []
        self.sampleIndex = 0
        self.trueLabel = list(map(str, raw_data.values[:, -2]))
        self.init_memory = self.show_info()
        self.micro_balls = self.init()

    @staticmethod
    def normalized(data):
        scaler = MinMaxScaler(feature_range=(0, 1))  # 数据缩放
        temp = data.values[:, :-2]
        print(len(temp[0]))
        if len(temp[0]) > 2:
            pca = PCA(n_components=2)
            data = pca.fit_transform(temp)
            data = scaler.fit_transform(data)
        else:
            data = scaler.fit_transform(temp)
        return data

    @staticmethod
    def get_nearest_micro_ball(sample, micro_balls):
        smallest_distance = sys.float_info.max
        nearest_micro_ball = None
        nearest_micro_ball_index = -1
        for i, micro_ball in enumerate(micro_balls):
            current_distance = np.linalg.norm(micro_ball.center - sample) - micro_ball.radius
            if current_distance < smallest_distance:
                smallest_distance = current_distance
                nearest_micro_ball = micro_ball
                nearest_micro_ball_index = i
        if nearest_micro_ball is None:
            print("nearest_micro_ball is None")
        return nearest_micro_ball_index, nearest_micro_ball, smallest_distance

    def init(self):
        init_time0 = time.perf_counter()
        init_data = self.data[self.sampleIndex: self.sampleIndex + self.velocity]
        init_label = self.trueLabel[self.sampleIndex: self.sampleIndex + self.velocity]
        init_point = [DataPoint(data, label) for data, label in zip(init_data, init_label) if label != "nan"]

        self.sampleIndex += self.velocity
        mb_arr, gb_obj = main(np.array(init_point))

        data = []
        for arr in mb_arr:
            for i in arr:
                data.append(i)
        init_mb_list = [MicroBall(mb) for mb in mb_arr]
        self.hb_evaluate(gb_obj, t=0, mb=len(mb_arr), start_time=init_time0)
        print("now have", len(mb_arr), "micro-balls")
        return init_mb_list

    def fit_predict(self):
        t = 1
        while self.sampleIndex < len(self.data):
            if self.sampleIndex == len(self.data):
                break
            fit_time0 = time.perf_counter()
            for i in range(self.velocity):
                if self.sampleIndex == len(self.data):
                    break
                if not self.micro_balls:
                    print("self.micro_balls is None")
                new_data = self.data[self.sampleIndex]
                new_label = self.trueLabel[self.sampleIndex]
                if new_label == "nan":
                    self.sampleIndex += 1
                    continue
                new_point = DataPoint(new_data, new_label, t)
                self.sampleIndex += 1
                centers = np.array([mb.center for mb in self.micro_balls])
                kd_tree = KDTree(centers)
                distances, indexes = kd_tree.query(new_point.data, k=2)
                MGBo, MGBp = self.micro_balls[indexes[0]], self.micro_balls[indexes[1]]
                if MGBo.DM < MGBp.DM:
                    MGB_DM_min, MGB_DM_min_index = MGBo, indexes[0]
                else:
                    MGB_DM_min, MGB_DM_min_index = MGBp, indexes[1]
                norm_MGBo_p = np.linalg.norm(MGBo.center - new_point.data)
                norm_MGBp_p = np.linalg.norm(MGBp.center - new_point.data)
                if norm_MGBo_p <= MGBo.radius and norm_MGBp_p <= MGBp.radius:
                    # print("case 2")
                    try:
                        insert = MGB_DM_min.insert_point(new_point)
                        if insert:
                            del self.micro_balls[MGB_DM_min_index]
                            self.micro_balls.extend(insert)
                    except Exception as e:
                        print(e)
                elif norm_MGBo_p > MGBo.radius and norm_MGBp_p > MGBp.radius:
                    dist, index = kd_tree.query(MGBo.center, k=2)
                    # MIR_o = (dist[1] * MGBo.radius) / (MGBo.radius + self.micro_balls[index[1]].radius)
                    MIR_o = dist[1] - self.micro_balls[index[1]].radius
                    if norm_MGBo_p <= MIR_o:
                        # print("case 3")
                        try:
                            insert = MGBo.insert_point(new_point)
                            if insert:
                                del self.micro_balls[indexes[0]]
                                self.micro_balls.extend(insert)
                        except Exception as e:
                            print(e)
                    else:
                        # print("case 4")
                        self.micro_balls.append(MicroBall([new_point]))
                else:
                    # print("case 1")
                    if norm_MGBp_p > MGBp.radius:
                        try:
                            insert = MGBo.insert_point(new_point)
                            if insert:
                                del self.micro_balls[indexes[0]]
                                self.micro_balls.extend(insert)
                        except Exception as e:
                            print(e)
                        else:
                            try:
                                insert = MGBp.insert_point(new_point)
                                if insert:
                                    del self.micro_balls[indexes[1]]
                                    self.micro_balls.extend(insert)
                            except Exception as e:
                                print(e)
            np_end_time = time.perf_counter()
            print("np_time = ", np_end_time - fit_time0)
            w_start_time = time.perf_counter()
            for i in range(len(self.micro_balls)):
                changed_flag = False
                n = self.micro_balls[i].num
                m = len([p for p in self.micro_balls[i].points if p.t == t])
                sum_w_dic = {t: sum(p.w for p in ps) for t, ps in
                             groupby(sorted(self.micro_balls[i].points, key=lambda x: x.t), key=lambda x: x.t)}
                # print(sum_w_dic)
                self.threshold = {t: (sum_w_dic[t]) / (n * max(1, m)) for t in sum_w_dic}
                # print(self.threshold)
                for p in self.micro_balls[i].points:
                    if p.t != t:
                        p.w = (((n - m) / (n * (t - p.t))) ** (t - p.t)) * p.w
                    if p.w <= self.threshold[p.t]: changed_flag = True
                if changed_flag:
                    temp_points = [p for p in self.micro_balls[i].points if p.w > self.threshold[p.t]]
                    self.micro_balls[i] = MicroBall(np.array(temp_points))
            temp_mb = [mb for mb in self.micro_balls if len(mb.points) != 0]
            self.micro_balls = temp_mb
            w_end_time = time.perf_counter()
            print("w_time = ", w_end_time - w_start_time)

            connect_start_time = time.perf_counter()
            clusters = self.connect()
            connect_end_time = time.perf_counter()
            print("connect_time = ", connect_end_time - connect_start_time)
            evaluate_start_time = time.perf_counter()
            self.hb_evaluate(clusters, t=t, start_time=fit_time0)
            evaluate_end_time = time.perf_counter()
            print("evaluate_time = ", evaluate_end_time - evaluate_start_time)
            t += 1
            print("now have", len(self.micro_balls), "micro-balls")

    def connect_old(self):
        mb_points_arr = [mb.points for mb in self.micro_balls]
        mb_data_arr = [mb.data for mb in self.micro_balls]
        radius = []
        for arr in mb_data_arr:
            if len(arr) >= 2:
                radius.append(get_radius(arr))
        radius_median = np.median(radius)
        radius_mean = np.mean(radius)
        radius_detect = max(radius_median, radius_mean)
        while True:
            ball_number_old = len(mb_points_arr)
            mb_points_arr = normalized_ball(mb_points_arr, radius_detect)  # 归一化
            mb_points_arr = np.array([a for a in mb_points_arr if len(a) != 0])
            ball_number_new = len(mb_points_arr)
            if ball_number_new == ball_number_old:
                break
        cluster = connect_ball(mb_points_arr, [], 2)
        data = []
        for points in mb_points_arr:
            for p in points:
                data.append(p.data)
        hb_plot(cluster, [])
        return cluster

    def connect(self):
        mb_points_arr = []
        for ball in self.micro_balls:
            mb_points_arr.extend(ball.points)
        _, cluster = main(np.array(mb_points_arr))
        return cluster

    @staticmethod
    def show_info():
        pid = os.getpid()
        p = psutil.Process(pid)
        memory = p.memory_info().rss / 1024 / 1024
        return memory

    def hb_evaluate(self, clusters, t, start_time, record=True, mb=-1):
        labelSet = set()
        clu = {}
        for cluster in clusters.values():
            if cluster.label not in labelSet:
                labelSet.add(cluster.label)
                clu[cluster.label] = {
                    "num": 0,
                    "label": []
                }
            clu[cluster.label]["num"] += cluster.num
            for p in cluster.points:
                clu[cluster.label]["label"].append(p.trueLabel)
        numOfClusters = len(labelSet)
        puritySum = 0
        numOfSamples = 0
        CdSum = 0
        for c in clu.values():
            numOfSamples += c["num"]
            a = max(c["label"], key=c["label"].count)
            Cd = c["label"].count(a)
            puritySum += Cd / c["num"]
            CdSum += Cd
        mean_purity = puritySum / numOfClusters
        accuracy = CdSum / numOfSamples
        print("==================" + "t = " + t.__str__() + "=======================")
        print("mean_purity\t", mean_purity)
        print("accuracy\t", accuracy)
        if record:
            with open('notebooks/' + self.datasetName, newline='', mode='a+') as f:
                writer = csv.writer(f)
                numOfMicroBalls = mb
                if mb == -1:
                    numOfMicroBalls = len(self.micro_balls)
                memory = self.show_info() - self.init_memory
                points_num = sum([cluster.num for cluster in clusters.values()])
                row = [t,
                       mean_purity, accuracy, memory,
                       numOfClusters, numOfMicroBalls,
                       points_num, time.perf_counter() - start_time
                       ]
                writer.writerow(row)
        return mean_purity, accuracy


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
        for key in gbs.keys():
            if gbs[key].label == list[i]:
                plt.scatter(gbs[key].data[:, 0], gbs[key].data[:, 1], s=4, c=color[i], linewidths=5, alpha=0.9,
                            marker='o', label=label_c[i])
                break

    for key in gbs.keys():
        for i in range(0, len(list)):
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


def start(data, dataset_name, v):
    M = MGBStream(data, dataset_name, velocity=v)
    M.fit_predict()
