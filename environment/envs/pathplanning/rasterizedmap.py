import copy

import numpy as np
import cv2 as cv
import os
import sys
import math

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../../PathPlanningAlgorithms/")

from Map.Color.Color import Color
from Map.Continuous.samplingmap import samplingmap

def sind(theta):
    return math.sin(theta / 180.0 * math.pi)

def cosd(theta):
    return math.cos(theta / 180.0 * math.pi)


class rasterizedmap:
    def __init__(self, _samplingmap: samplingmap, x_grid: int, y_grid: int):
        self.sampling_map = _samplingmap
        self.x_grid = x_grid                                                                            # x栅格数
        self.y_grid = y_grid                                                                            # y栅格数
        self.x_meter_per_grid = self.sampling_map.x_size / self.x_grid                                  # x每格对应的实际距离(米)
        self.y_meter_per_grid = self.sampling_map.y_size / self.y_grid                                  # y每格对应的实际距离(米)
        self.x_pixel_per_grid = self.sampling_map.pixel_per_meter * self.x_meter_per_grid               # x每格对应的实际长度(像素)
        self.y_pixel_per_grid = self.sampling_map.pixel_per_meter * self.y_meter_per_grid               # y每格对应的实际长度(像素)
        self.map_flag = [[0 for _ in range(x_grid)] for _ in range(y_grid)]

        self.image2 = np.zeros([self.sampling_map.width, self.sampling_map.height, 3], np.uint8)
        self.image2[:, :, 0] = np.ones([self.sampling_map.width, self.sampling_map.height]) * 255
        self.image2[:, :, 1] = np.ones([self.sampling_map.width, self.sampling_map.height]) * 255
        self.image2[:, :, 2] = np.ones([self.sampling_map.width, self.sampling_map.height]) * 255

        self.name4image = self.sampling_map.name4image + 'rasterized'

        self.map_rasterization()
        self.draw_rasterization_map()
        cv.imshow(self.name4image, self.image2)
        cv.waitKey(0)

    def is_grid_has_obs(self, points: list) -> int:
        for _point in points:
            if self.sampling_map.point_is_in_obs(_point):
                return 1
        '''四个顶点都不在障碍物里面'''

        assert len(points) == 4
        for i in range(4):
            if self.sampling_map.line_is_in_obs(points[i % 4], points[(i + 1) % 4]):
                return 1
        '''四个边都不在障碍物里面'''

        for _obs in self.sampling_map.obs:
            if _obs[0] == 'circle' or _obs[0] == 'ellipse':
                if self.sampling_map.point_is_in_poly(center=None, r=None, points=points, point=_obs[2]):
                    return 1
            else:
                if self.sampling_map.point_is_in_poly(center=None, r=None, points=points, point=[_obs[1][0], _obs[1][1]]):
                    return 1
        '''障碍物不在格子里面'''
        return 0

    def map_rasterization(self):
        for i in range(self.x_grid):
            for j in range(self.y_grid):
                rec = [[i * self.x_meter_per_grid, j * self.y_meter_per_grid],
                       [(i + 1) * self.x_meter_per_grid, j * self.y_meter_per_grid],
                       [(i + 1) * self.x_meter_per_grid, (j + 1) * self.y_meter_per_grid],
                       [i * self.x_meter_per_grid, (j + 1) * self.y_meter_per_grid]]
                self.map_flag[i][j] = self.is_grid_has_obs(rec)

    '''drawing'''

    def draw_rasterization_map(self):
        self.map_draw_gird_rectangle()
        self.map_draw_x_grid()
        self.map_draw_y_grid()
        self.map_draw_obs()
        self.map_draw_boundary()
        self.map_draw_start_terminal()

    def map_draw_gird_rectangle(self):
        for i in range(self.x_grid):
            for j in range(self.y_grid):
                if self.map_flag[i][j] == 1:
                    pt1 = self.grid2pixel(coord_int=[i, j], pos='left-bottom', xoffset=-0, yoffset=0)
                    pt2 = self.grid2pixel(coord_int=[i, j], pos='right-top', xoffset=0, yoffset=0)
                    cv.rectangle(self.image2, pt1, pt2, Color().LightGray, -1)

    def grid2pixel(self, coord_int: list, pos: str, xoffset=0, yoffset=0) -> tuple:
        """
        :brief:             to transfer grid in map to pixel in image
        :param coord_int:   coordinate [int, int]
        :param pos:         left-top, left-bottom, right-top. right-bottom
        :param xoffset:     xoffset
        :param yoffset:     yoffset
        :return:            pixel [int, int] (left-bottom)
        :tips:              the direction of offset is the same as that of image rather than real world or grid map
        """
        x = self.sampling_map.x_offset + coord_int[0] * self.x_pixel_per_grid
        y = self.sampling_map.height - self.sampling_map.y_offset - coord_int[1] * self.y_pixel_per_grid  # sef default to left-bottom

        if pos == 'left-bottom':
            return int(x) + xoffset, int(y) + yoffset
        elif pos == 'left-top':
            return int(x) + xoffset, int(y - self.y_pixel_per_grid) + yoffset
        elif pos == 'right-bottom':
            return int(x + self.x_pixel_per_grid) + xoffset, int(y) + yoffset
        elif pos == 'right-top':
            return int(x + self.x_pixel_per_grid) + xoffset, int(y - self.y_pixel_per_grid) + yoffset
        else:
            print('FUNCTION <grid2pixel>--ERROR input')
            return ()

    def map_draw_x_grid(self):
        for i in range(self.y_grid + 1):
            pt1 = self.grid2pixel(coord_int=[0, i], pos='left-bottom')
            pt2 = self.grid2pixel(coord_int=[self.x_grid, i], pos='left-bottom')
            cv.line(self.image2, pt1, pt2, Color().Black, 1)

    def map_draw_y_grid(self):
        for i in range(self.x_grid + 1):
            pt1 = self.grid2pixel(coord_int=[i, 0], pos='left-bottom')
            pt2 = self.grid2pixel(coord_int=[i, self.y_grid], pos='left-bottom')
            cv.line(self.image2, pt1, pt2, Color().Black, 1)

    def map_draw_obs(self):
        if self.sampling_map.obs is None:
            print('No obstacles!!')
            return
        for [name, constraints, pts] in self.sampling_map.obs:   # [name, [], [pt1, pt2, pt3]]
            if name == 'circle':
                cv.circle(self.image2, self.sampling_map.dis2pixel(pts), self.sampling_map.length2pixel(constraints[0]), Color().DarkGray, -1)
            elif name == 'ellipse':
                cv.ellipse(img=self.image2,
                           center=self.sampling_map.dis2pixel(pts),
                           axes=(self.sampling_map.length2pixel(constraints[0]), self.sampling_map.length2pixel(constraints[1])),
                           angle=-constraints[2],
                           startAngle=0.,
                           endAngle=360.,
                           color=Color().DarkGray,
                           thickness=-1)
            else:
                cv.fillConvexPoly(self.image2, points=np.array([list(self.sampling_map.dis2pixel(pt)) for pt in pts]), color=Color().DarkGray)

    def map_draw_boundary(self):
        cv.rectangle(self.image2, self.sampling_map.dis2pixel([0., 0.]), self.sampling_map.dis2pixel([self.sampling_map.x_size, self.sampling_map.y_size]), Color().Black, 2)

    def map_draw_start_terminal(self):
        if self.sampling_map.start and self.sampling_map.terminal:
            cv.circle(self.image2, self.sampling_map.dis2pixel(self.sampling_map.start), 5, Color().Red, -1)
            cv.circle(self.image2, self.sampling_map.dis2pixel(self.sampling_map.terminal), 5, Color().Blue, -1)
        else:
            print('No start point or terminal point')

    '''drawing'''

    def point_in_grid(self, point: list) -> list:
        if self.sampling_map.point_is_out(point):
            return [-1, -1]

        return [int(point[0] / self.x_meter_per_grid), int(point[1] / self.y_meter_per_grid)]

    def is_grid_available(self, grid: list) -> bool:
        return True if self.map_flag[grid[0]][grid[1]] == 0 else False
