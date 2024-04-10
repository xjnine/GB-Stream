import math

import numpy as np


class DataPoint:
    def __init__(self, data, label, t=0, w=1):
        self.data = data
        self.trueLabel = label
        self.t = t
        self.w = w


class MicroBall:
    def __init__(self, points):
        self.points = points
        self.data = np.array([p.data for p in points])
        self.num = self.get_num()
        self.center = self.get_center()
        self.radius = self.get_radius()
        self.DM = self.get_dm()

    def get_radius(self):
        if self.num == 0: return None
        return max(((self.data - self.center) ** 2).sum(axis=1) ** 0.5)

    def get_center(self):
        if self.num == 0: return None
        return self.data.mean(0)

    def get_num(self):
        return len(self.data)

    def get_dm(self):
        if self.num == 0: return None
        diffMat = np.tile(self.center, (self.num, 1)) - self.data
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5
        sum_radius = 0
        for i in distances:
            sum_radius += i
        DM = sum_radius / self.num
        return DM

    def insert_point(self, point):
        self.points = np.append(self.points, np.array([point]), axis=0)
        self.data = np.append(self.data, np.array([point.data]), axis=0)
        self.center = self.get_center()
        self.radius = self.get_radius()
        self.num = self.get_num()
        self.DM = self.get_dm()
        return self.is_division()

    def is_division(self):
        if math.isclose(self.radius, 0, rel_tol=1e-10) or self.num < 8:
            return False
        else:
            split = self.spilt_ball()
            if split:
                ball_1, ball_2 = split
                DM_parent = self.DM
                DM_child_1 = ball_1.DM
                DM_child_2 = ball_2.DM
                t1 = ((DM_child_1 < DM_parent) and (DM_child_2 < DM_parent))
                if t1:
                    return [ball_1, ball_2]
                else:
                    return False
            else:
                return False

    def spilt_ball(self):
        data = self.data
        if len(np.unique(data, axis=0)) == 1:
            return False
        point1 = []
        point2 = []
        n, m = data.shape
        X = data.T
        G = np.dot(X.T, X)
        H = np.tile(np.diag(G), (n, 1))
        D = np.sqrt(np.abs(H + H.T - G * 2))
        r, c = np.where(D == np.max(D))
        r1 = r[1]
        c1 = c[1]
        if r1 == c1:
            return False
        for j in range(0, len(data)):
            if D[j, r1] < D[j, c1]:
                point1.extend([self.points[j]])
            else:
                point2.extend([self.points[j]])
        if len(point1) * len(point2) == 0:
            return False
        else:
            point1 = np.array(point1)
            point2 = np.array(point2)
            return [MicroBall(points=point1), MicroBall(points=point2)]
