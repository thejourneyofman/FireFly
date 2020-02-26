#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2020 Chao (Chase) Xu
#
"""A firefly graph is an optimized data structure that uses graph and geometry
theories to find the densest subsets of the time series and spatial datasets,
especially those to be more like some kind of sparse or outliers' data.
There are three classes in `Firefly Graph`:
- `Footprint` - Returns a class of a single node named as Footprint in the Graph. Each node represents a footprint of fireflies at a certain point in timeline.
- `Resort` - Returns a class of a resort or a group of footprints in the Graph. Each resort represents a group of footprints of fireflies.
- `Cruise` - Returns a class of a graph. Each Cruise represents a group of resorts where fireflies have toured.
"""

import numpy as np
from math import sqrt, acos, pi
from collections import deque
from itertools import product
from functools import reduce

class Footprint():
    u"""Returns a class of a single node named as Footprint in the Graph.
        Each node represents a footprint of fireflies at a certain point in timeline.
       :param uid: the id of each footprint which should be unique for the entire timeline.
       :param timestamp: the timestamp or time series No of the footprint in timeline.
       :param addresstamp: the point(X,Y) set of each node that represents a firefly's footprint.
       :param popularity: the size of each node or footprint that contains all none-zero pixels.
       :param label: the label of each footprint using its timestamp, center (X, Y) and popularity,
                      that can make sure it's also unique.
       :param # add param for future use
    """
    def __init__(self, uid, timestamp, addresstamp):
        self.uid = uid
        self.timestamp = str(timestamp)
        self.addresstamp = addresstamp
        self.popularity = len(addresstamp)
        self._boundRect()
        self.label = '_'.join([self.timestamp, str(int(self.center[0])), str(int(self.center[1])), str(self.popularity)])

    def _boundRect(self):
        u"""Find the enclosing rectangle of each footprint and extact the key points including all the
            corners' and center's (X, Y).
            :param # add param for future use
        """
        self.upperleft = list(map(min, zip(*self.addresstamp)))
        self.bottomright = list(map(max, zip(*self.addresstamp)))
        self.upperright = [self.bottomright[0], self.upperleft[1]]
        self.bottomleft = [self.upperleft[0], self.bottomright[1]]
        (self.width, self.height) = (self.upperright[0] - self.bottomleft[0], self.bottomleft[1] - self.upperright[1])
        assert self.width >= 0
        assert self.height >= 0
        self.center = [self.upperleft[0] +  self.width / float(2), self.upperleft[1] + self.height / float(2)]
        self.corners = [self.upperright, self.bottomleft, self.upperleft, self.bottomright]

    def iter_edges(self):
        u"""a geometry method that iterate the four edges of each rectangle.
            :param # add param for future use
        """
        yield self.upperleft, self.upperright
        yield self.upperright, self.bottomright
        yield self.bottomright, self.bottomleft
        yield self.bottomleft, self.upperleft

    def is_point_in(self, point):
        u"""a geometry method that returns the result as true if a single point is inside a rectangle.
            noted: it also returns true if the single point just falls on one of the edges.
            :param point:a single point with its cordinate as (X, Y).
        """
        return (self.upperleft[0] <= point[0] <= self.upperright[0] and self.upperleft[1] <= point[1] <= self.bottomleft[1])

    def is_overlappedFootprint(self, footprint):
        u"""a geometry method that returns the result as true if two nodes or footprints overlapped in a certain frame.
            noted: it returns false if another footprint is just a point or a line, whose area size is zero but it returns
                   true if two rectangles overlapped at only one edge, where the overlapped area size can be zero.
            :param footprint:another node or footprint.
        """
        if footprint.width == 0 or footprint.height == 0 or footprint.popularity <= 1:
            return False
        for corner in footprint.corners:
            if self.is_point_in(corner):
                return True
        for corner in self.corners:
            if footprint.is_point_in(corner):
                return True
        return False

    def is_overlappedResort(self, resort):
        u"""a geometry method that returns the result as true if each node or footprint overlapped with a resort,
            in a certain frame, where a resort is a group of footprints that represent a tracked tour of a firefly.
            noted: it returns false if another footprint is just a point or a line, whose area size is zero but it
                    returns true if two rectangles overlapped at one edge, where the overlapped area size can be zero.
            :param resort:another group of footprints or a resort.
        """
        for corner in resort.corners:
            if self.is_point_in(corner):
                return True
        for corner in self.corners:
            if resort.is_point_in(corner):
                return True
        if self.intersection_area(resort) > 0:
            return True
        return False

    def intersection_area(self, footprint):
        u"""a geometry method that returns the overlapped area size of two footprints.
            :param footprint:another footprint.
        """
        dx = min(self.upperright[0], footprint.upperright[0]) - max(self.upperleft[0], footprint.upperleft[0])
        dy = min(self.bottomright[1], footprint.bottomright[1]) - max(self.upperright[1], footprint.upperright[1])
        #assert dx >= 0
        #assert dy >= 0
        return dx * dy

    def distance_from_resort(self, resort):
        u"""a geometry method that returns the shortest distance from a single footprint to a resort.
            :param resort:another group of footprints or a resort.
        """
        if self.is_overlappedResort(resort):
            return 0
        if self.center ==  resort.center:
            return 0
        if self._isLine():
            return 0
        line = (self.center, resort.center)
        edge1 = None
        edge2 = None
        if self._isPoint():
            return 0
        elif self._isLine():
            if self.width == 0:
                edge1 = self.bottomleft, self.upperleft
            elif self.height == 0:
                edge1 = self.upperleft, self.upperright
        else:
            for edge in self.iter_edges():
                if self._lines_intersect(edge, line):
                    edge1 = edge
                    break
        if resort._isPoint():
            return 0
        elif resort._isLine():
            if resort.width == 0:
                edge2 = resort.bottomleft, resort.upperleft
            elif self.height == 0:
                edge2 = resort.upperleft, resort.upperright
        else:
            for edge in resort.iter_edges():
                if self._lines_intersect(edge, line):
                    edge2 = edge
                    break
        assert edge1
        assert edge2
        distances = [
            self.distance_between_edge_and_point(edge1, edge2[0]),
            self.distance_between_edge_and_point(edge1, edge2[1]),
            self.distance_between_edge_and_point(edge2, edge1[0]),
            self.distance_between_edge_and_point(edge2, edge1[1]),
        ]

        return min(distances)

    def _isLine(self):
        u"""a geometry method that returns true if a footprint is in a line.
            :param # add param for future use
        """
        return (self.width == 0 and self.height > 1) or (self.height == 0 and self.width > 1)

    def _isPoint(self):
        u"""a geometry method that returns true if a footprint is a single point.
            :param # add param for future use
        """
        return (self.width == 0 and self.height == 1) or (self.height == 0 and self.width == 1)

    def _lines_intersect(self, line1, line2):
        u"""a geometry method that returns true if two lines intersect.
            :param line1:a tupple of two points.
            :param line2:a tupple of two points.
        """
        return self._lines_overlap_on_x_axis(line1, line2) and self._lines_overlap_on_y_axis(line1, line2)

    def _lines_overlap_on_x_axis(self, line1, line2):
        u"""a geometry method that returns true if two lines ovelap on X axis.
            :param line1:a tupple of two points.
            :param line2:a tupple of two points.
        """
        x1, x2, = line1[0][0], line1[1][0]
        x3, x4, = line2[0][0], line2[1][0]
        e1_left, e1_right = min(x1, x2), max(x1, x2)
        e2_left, e2_right = min(x3, x4), max(x3, x4)
        return (e1_left >= e2_left and e1_left <= e2_right) or (e1_right >= e2_left and e1_right <= e2_right) or \
               (e2_left >= e1_left and e2_left <= e1_right) or (e2_right >= e1_left and e2_right <= e1_right)

    def _lines_overlap_on_y_axis(self, line1, line2):
        u"""a geometry method that returns true if two lines ovelap on Y axis.
            :param line1:a tupple of two points.
            :param line2:a tupple of two points.
        """
        y1, y2, = line1[0][1], line1[1][1]
        y3, y4, = line2[0][1], line2[1][1]
        e1_top, e1_bot = min(y1, y2), max(y1, y2)
        e2_top, e2_bot = min(y3, y4), max(y3, y4)
        return (e1_top >= e2_top and e1_top <= e2_bot) or (e1_bot >= e2_top and e1_bot <= e2_bot) or \
               (e2_top >= e1_top and e2_top <= e1_bot) or (e2_bot >= e1_top and e2_bot <= e1_bot)

    def distance_between_edge_and_point(self, edge, point):
        u"""a geometry method that returns shortest distance froma a single point to the edge that is facing.
            :param edge:a tupple of two points.
            :param point:a (X, Y) cordinate.
        """
        if self._point_faces_edge(edge, point):
            area = self._triangle_area_at_points(edge[0], edge[1], point)
            base = sqrt((edge[0][0] - edge[1][0]) ** 2 + (edge[0][1] - edge[1][1]) ** 2)
            height = area / (0.5 * base)
            return height
        return min(sqrt((edge[0][0] - point[0]) ** 2 + (edge[0][1] - point[1]) ** 2),
                   sqrt((edge[1][0] - point[0]) ** 2 + (edge[1][1] - point[1]) ** 2))

    def _point_faces_edge(self, edge, point):
        u"""a geometry method that returns true if a single point is facing an edge.
            :param edge:a tupple of two points.
            :param point:a (X, Y) cordinate.
        """
        a = sqrt((edge[0][0] - edge[1][0]) ** 2 + (edge[0][1] - edge[1][1]) ** 2)
        b = sqrt((edge[0][0] - point[0]) ** 2 + (edge[0][1] - point[1]) ** 2)
        c = sqrt((edge[1][0] - point[0]) ** 2 + (edge[1][1] - point[1]) ** 2)
        ang1, ang2 = self._angle(b, a, c), self._angle(c, a, b)
        if ang1 > pi / 2 or ang2 > pi / 2:
            return False
        return True

    def _triangle_area_at_points(self, p1, p2, p3):
        u"""a geometry method that returns the area size of a triangle.
            :param p1, p2, p3: three (X, Y) cordinates of a triangle.
        """
        a = sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        b = sqrt((p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2)
        c = sqrt((p1[0] - p3[0]) ** 2 + (p1[1] - p3[1]) ** 2)
        s = (a + b + c) / float(2)
        area = sqrt(s * (s - a) * (s - b) * (s - c))
        return area

    def _angle(self, a, b, c):
        u"""a geometry method that returns angle using cos law.
        """
        divid = (a ** 2 + b ** 2 - c ** 2)
        divis = (2 * a * b)
        if (divis) > 0:
            result = float(divid) / divis
            if result <= 1.0 and result >= -1.0:
                return acos(result)
            return 0
        else:
            return 0

class Resort():
    u"""Returns a class of a resort or a group of footprints in the Graph.
        Each resort represents a group of footprints of fireflies.
       :param footprints: the group of footprints.
       :param # add param for future use
    """

    def __init__(self, footprints):
        self.footprints = footprints
        self._boundRect()

    def _boundRect(self):
        u"""Find the enclosing rectangle of each resort and extact the key points including all the
            corners' and center's cordinate (X, Y).
            :param # add param for future use
        """
        addresstamp = reduce(lambda x, y: x + y, [v.addresstamp for v in self.footprints])
        self.upperleft = list(map(min, zip(*addresstamp)))
        self.bottomright = list(map(max, zip(*addresstamp)))
        self.upperright = [self.bottomright[0], self.upperleft[1]]
        self.bottomleft = [self.upperleft[0], self.bottomright[1]]
        (self.width, self.height) = (self.upperright[0] - self.bottomleft[0], self.bottomleft[1] - self.upperright[1])
        assert self.width >= 0
        assert self.height >= 0
        self.center = [self.upperleft[0] + self.width / float(2), self.upperleft[1] + self.height / float(2)]
        self.corners = [self.upperright, self.bottomleft, self.upperleft, self.bottomright]

    def iter_edges(self):
        u"""a geometry method that iterate the four edges of each rectangle.
            :param # add param for future use
        """
        yield self.upperleft, self.upperright
        yield self.upperright, self.bottomright
        yield self.bottomright, self.bottomleft
        yield self.bottomleft, self.upperleft

    def addFootprint(self, footprint):
        u"""Add a footprint to the resort.
            :param footprint: a node or footprint
        """
        self.footprints.append(footprint)

    def getLastFootprint(self):
        u"""Returns the latest footprint in the timeline.
            :param # add param for future use
        """
        if self.footprints is None:
            return
        return self.footprints[-1]

    def is_point_in(self, point):
        u"""a geometry method that returns the result as true if a single point is inside a rectangle.
            noted: it also returns true if the single point just falls on one of the edges.
            :param point:a single point with its cordinate as (X, Y).
        """
        return (self.upperleft[0] <= point[0] <= self.upperright[0] and self.upperleft[1] <= point[1] <= self.bottomleft[1])

    def _isLine(self):
        u"""a geometry method that returns true if a group of footprints or a resort is in a line.
            :param # add param for future use
        """
        return (self.width == 0 and self.height > 1) or (self.height == 0 and self.width > 1)

    def _isPoint(self):
        u"""a geometry method that returns true if a group of footprints or a resort is a single point.
            :param # add param for future use
        """
        return (self.width == 0 and self.height == 1) or (self.height == 0 and self.width == 1)

class Cruise():
    u"""Returns a class of a Graph.
        Each Cruise represents a group of resorts where fireflies have toured.
       :param V: the group of all footprints
       :param E: the group of all edges between two footprints
       :param resorts: the group of all resorts
       :param temp_component: a temporary list to store footprints for depth-first search
       :param source: the unique index of footprints
       :param upperleft: the upperleft corner of the bounding rectangle of the graph
       :param bottomright: the bottomright corner of the bounding rectangle of the graph
       :param upperright: the upperright corner of the bounding rectangle of the graph
       :param bottomleft: the bottomleft corner of the bounding rectangle of the graph
    """
    def __init__(self):
        self.V = []
        self.E = []
        self.resorts = {}
        self.neighbours = {}
        self.temp_component = []
        self.source = 0
        self.upperleft = -1
        self.bottomright = -1
        self.upperright = -1
        self.bottomleft = -1

    def getFootprintId(self):
        u"""Return the id list of all the footprints.
            :param res: the node started searching by dfs.
        """
        idList = []
        for v in self.V:
            idList.append(v.uid)
        return idList

    def recursive_findDenseZones(self, neighbors):
        u"""A recursive way using greedy search to find and extracting densest sub graphs.
            :param neighbors: a dictionary to store the neighbourhood pairs
        """
        try:
            cruisingSpots = self.greedy_degree_density(neighbors)
            yield cruisingSpots
            neighbors = dict(filter(lambda x: x[0] not in cruisingSpots.getFootprintId(), neighbors.items()))
            for key, value in neighbors.items():
                neighbors[key] = list(filter(lambda x: x not in cruisingSpots.getFootprintId(), value))
            if any(neighbors.values()):
                for ele in list(self.recursive_findDenseZones(neighbors)):
                    yield ele
        except StopIteration:
            raise KeyError(neighbors)

    def check_collision(self, footprint):
        u"""a geometry method that returns the result as true if a single foortprint is inside the
            bouding rectangle of the graph.
            :param footprint:a single footprint with its four courners.
        """
        return self.upperleft[0] < footprint.upperleft[0] < footprint.upperright[0] < self.upperright[0] and \
               self.upperleft[1] < footprint.upperleft[1] < footprint.bottomleft[1] < self.bottomleft[1]

    def addGrids(self, grid, label):
        u"""A dynamic way to add a new (X,Y) grid which is a 2D frame in a certain point in timeline.
            It's a method to load new frames and generate the graph by adding it dynamically.
            :param grid:a 2D frame in a certain point in timeline.
            :param label:a label that will be used for the footprints.
        """
        self.grid = grid
        self.label = label
        self.H = len(grid)
        self.W = len(grid[0])
        self.visited = [[False] * self.W for _ in range(self.H)]
        self.clusters = [self._resort_size(r, c) for r, c in product(range(self.H), range(self.W)) if self.grid[r][c] != 0 and not self.visited[r][c]]
        self.subsets = []

        assert len(self.V) == len(self.neighbours.keys())

        if len(self.V) == 0  or not any(self.neighbours.values()):
            for addresstamp in self.clusters:
                footprint = Footprint(self.source, label, addresstamp)
                self.source += 1
                self.neighbours[footprint.uid] = []
                for v in self.V:
                    if v.is_overlappedFootprint(footprint):
                        self.E.append((v, footprint))
                        self.neighbours.get(v.uid).append(footprint.uid)
                        self.neighbours.get(footprint.uid).append(v.uid)
                        self.resorts[v] = Resort([footprint])
                self.V.append(footprint)
            return

        connected_footprints = []
        for sub in self.recursive_findDenseZones(self.neighbours):
            self.subsets.append(sub)
            connected_footprints.extend(sub.getFootprintId())
        for addresstamp in self.clusters:
            footprint = Footprint(self.source, label, addresstamp)
            self.source += 1
            self.neighbours[footprint.uid] = []
            if len(addresstamp) == 1:
                self.V.append(footprint)
                continue
            new_sub = 1
            for sub in self.subsets:
                if sub.check_collision(footprint):
                    new_sub = 0
                    new_resort = 1
                    for key, resort in sub.resorts.items():
                        if resort.getLastFootprint().is_overlappedFootprint(footprint):
                            new_resort = 0
                            for n in resort.footprints:
                                self.E.append((n, footprint))
                                self.neighbours.get(n.uid).append(footprint.uid)
                                self.neighbours.get(footprint.uid).append(n.uid)
                            sub.resorts[key].addFootprint(footprint)
                            self.resorts[key].addFootprint(footprint)
                            break
                    if new_resort:
                        sub.resorts[footprint] = Resort([footprint])
                        self.resorts[footprint] = Resort([footprint])
                    break
            if new_sub:
                min_distance = sqrt(self.H**2 + self.W**2)
                closest = None
                for key, resort in self.resorts.items():
                    if footprint.distance_from_resort(resort) <  min_distance:
                        min_distance = footprint.distance_from_resort(resort)
                        closest = key
                assert  closest
                for n in self.resorts[closest].footprints:
                    self.E.append((n, footprint))
                    self.neighbours.get(n.uid).append(footprint.uid)
                    self.neighbours.get(footprint.uid).append(n.uid)
            self.V.append(footprint)

    def greedy_degree_density(self, neighbors):
        u"""Returns a sub graph or cruise with the optimal degree density using
            Charikar's greedy algorithm.
            :param neighbors: a dictionary to store the neighbourhood pairs
        """
        degrees = {key: len(value) for key, value in neighbors.items() if len(value) > 0}
        sum_degrees = sum(degrees.values())
        num_footprints = len(degrees.keys())
        footprints = sorted(degrees, key=degrees.get)
        bin_boundaries = [0]
        curr_degree = 0
        for i, v in enumerate(footprints):
            if degrees[v] > curr_degree:
                bin_boundaries.extend([i] * (degrees[v] - curr_degree))
                curr_degree = degrees[v]
        footprint_pos = dict((v, pos) for pos, v in enumerate(footprints))
        nbrs = dict((v, set(neighbors[v])) for v in footprints)
        max_degree_density = sum_degrees / float(num_footprints)
        ind = 0

        for v in footprints:
            num_footprints -= 1
            while degrees[v] > 0:
                pos = footprint_pos[v]
                bin_start = bin_boundaries[degrees[v]]
                footprint_pos[v] = bin_start
                footprint_pos[footprints[bin_start]] = pos
                footprints[bin_start], footprints[pos] = footprints[pos], footprints[bin_start]
                bin_boundaries[degrees[v]] += 1
                degrees[v] -= 1

            for u in list(nbrs[v]):
                nbrs[u].remove(v)
                pos = footprint_pos[u]
                bin_start = bin_boundaries[degrees[u]]
                footprint_pos[u] = bin_start
                footprint_pos[footprints[bin_start]] = pos
                footprints[bin_start], footprints[pos] = footprints[pos], footprints[bin_start]
                bin_boundaries[degrees[u]] += 1
                degrees[u] -= 1
                sum_degrees -= 2

            if num_footprints > 0:
                current_degree_density = sum_degrees / float(num_footprints)
                if current_degree_density > max_degree_density:
                    max_degree_density = current_degree_density
                    ind = len(footprints) - num_footprints

        optimal_footprints = footprints[ind:]
        return self.subCruise(optimal_footprints)

    def subCruise(self, footprints):
        u"""Returns a class of a subset of the Cruise, or a sub graph.
            Each sub cruise represents a group of resorts where fireflies have toured.
            :param footprints: the group of all footprints in the subset
        """
        sub = Cruise()
        sub.V = [v for v in self.V if v.uid in footprints]
        sub.E = []
        sub.neighbours = {}
        sub.resorts = {}
        for key, value in self.neighbours.items():
            sub.neighbours[key] = []
            if key in footprints:
                sub.V.extend([self.V[v] for v in value if self.V[v] not in sub.V])
                sub.E.extend([(key, v) for v in value])
                sub.neighbours.get(key).extend(value)
                for v in value:
                    sub.neighbours[v] = [key]

        p = set(sub.V)
        #sorted_resorts = sorted(self.resorts, key=len, reverse=True)
        for key, resort in self.resorts.items():
            s = set(resort.footprints)
            if len(tuple(s & p)) > 0:
                sub.resorts[key] = Resort(list(tuple(s & p)))
                p -= s
                if len(tuple(p)) == 0:
                    break

        addresstamp = reduce(lambda x, y: x + y, [v.addresstamp for v in sub.V])
        sub.upperleft = list(map(min, zip(*addresstamp)))
        sub.bottomright = list(map(max, zip(*addresstamp)))
        sub.upperright = [sub.bottomright[0], sub.upperleft[1]]
        sub.bottomleft = [sub.upperleft[0], sub.bottomright[1]]
        (sub.W, sub.H) = (sub.upperright[0] - sub.bottomleft[0], sub.bottomleft[1] - sub.upperright[1])
        sub.grid = self.grid[np.ix_(list(range(sub.upperleft[1],sub.bottomleft[1])), list(range(sub.upperleft[0],sub.bottomright[0])))]

        assert sub.W >= 0
        assert sub.H >= 0
        return sub

    def _neighbors(self, r, c):
        u"""a geometry method that generates the neighborhood pairs of the given rows and columns
            which are inside the grids.Each cordinate has 8 neighbors including the horizontal vertical
            and diagonal ones;
            :param r,c: a neighborhood pair in the grid
        """
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (1, 0), (0, -1), (0, 1), (1, 1), (1, -1)]:
            if (0 <= r + dr < self.H) and (0 <= c + dc < self.W):
                yield r + dr, c + dc

    def _resort_size(self, r, c):
        u"""Returns the area size of the resort connected to the given neighborhood coordinate and
            0 if the coordinate is not in the resort or it has been explored in a previous call.
            :param r,c: a neighborhood pair in the grid
        """
        areasize = 1
        stack = [(r, c)]
        self.visited[r][c] = True
        resort = [(c,r)]
        while stack:
            for r, c in self._neighbors(*stack.pop()):
                if self.grid[r][c] and not self.visited[r][c]:
                    stack.append((r, c))
                    self.visited[r][c] = True
                    areasize += 1
                    resort.append((c,r))
        return resort

    def _dfsearch_recursive(self, footprint):
        u"""A recursive way using depth-first algorithm to search nodes or footprints by
            their neighbours to return the connected list.
            :param footprint: the footprint started searching by dfs.
        """
        self.visited[footprint] = 1
        self.temp_component.append(footprint)
        for neighbour in self.neighbours[footprint]:
            if self.visited[neighbour] == 0:
                self._dfsearch(neighbour)

    def _dfs_non_recursive(self, footprints):
        u"""A None recursive way using depth-first algorithm to search neighbours
            and return a full connected list.
            :param footprints: a group of the footprints started searching by dfs.
        """
        visited = {}
        for v in footprints:
            visited[v] = False
        cluster = []
        end_of_scan = footprints[0]
        for v in footprints:
            if not any(x != True for x in visited.values()) and cluster:
                cluster.append(end_of_scan)
                self.resorts.append(cluster)
                break
            if not visited[v]:
                yield v
                visited[v] = True
                stack = [(v, iter(self.neighbours[v]))]
                if v != end_of_scan and cluster:
                    cluster.append(end_of_scan)
                    self.resorts.append(cluster)
                    end_of_scan = 0
                    cluster = []
                while stack:
                    _, neighbourlist = stack[-1]
                    try:
                        neighbour = next(neighbourlist)
                        if not visited[neighbour]:
                            yield neighbour
                            visited[neighbour] = True
                            stack.append((neighbour, iter(self.neighbours[neighbour])))
                            cluster.append(neighbour)
                    except StopIteration:
                        end_of_scan = v
                        stack.pop()

    def updateResorts(self):
        u"""Generate a list of all connected components using a
            None recursive dfs algorithm from all edges pairs
            :param # add param for future use
        """
        self.resorts = []
        for x in self._dfs_non_recursive(self.V):
            pass
            #print("visited", x)


    def bfsearch(self, start_footprint):
        u"""A breadth-first algorithm to search all connected components from
            a start footprint or a node. It returns all the visited noces and the edge
            degree.
            :param start_footprint: the footprint started searching by bfs.
        """
        queue = deque([start_footprint])
        visited_footprints = {}
        hop_away = 0
        while len(queue) != 0:
            label = queue.popleft()
            if label not in visited_footprints:
                visited_footprints[label] = [len(self.neighbours[label]), hop_away]
                queue += [x for x in self.neighbours[label] if x not in visited_footprints.keys()]
                hop_away += 1
        return visited_footprints




