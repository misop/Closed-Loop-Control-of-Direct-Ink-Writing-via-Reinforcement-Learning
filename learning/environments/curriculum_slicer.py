#region Imports

import numpy as np
import random
import math
import collections
import warnings
from PIL import Image
from PIL import ImageDraw
from hilbert import decode
from itertools import chain 
import pyclipper
import triangle as tr
import meshcut
from astar import AStar
import trimesh

#endregion

#region Pathfinder

class PathFinder(AStar):
    def __init__(self, points, graph, triangles):
        self.points = points
        self.graph = graph
        # prepare start and end points
        se = np.array([[0,0], [0,0]])
        self.points = np.vstack((self.points, se))
        self.start_idx = self.points.shape[0]-2
        self.end_idx = self.points.shape[0]-1
        self.graph[self.start_idx] = dict()
        self.graph[self.end_idx] = dict()
        self.triangles = triangles

    def heuristic_cost_estimate(self, n1, n2):
        P = self.points[n1, :]
        Q = self.points[n2, :]
        return math.hypot(P[0] - Q[0], P[1] - Q[1]) / 10

    def distance_between(self, n1, n2):
        return self.graph[n1][n2]

    def neighbors(self, node):
        return self.graph[node].keys()

    def sign(self, P, Q, R):
        return (P[0] - R[0]) * (Q[1] - R[1]) - (Q[0] - R[0]) * (P[1] - R[1])

    def point_in_triangle(self, P, A, B, C):
        d1 = self.sign(P, A, B)
        d2 = self.sign(P, B, C)
        d3 = self.sign(P, C, A)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def set_point(self, idx, P):
        # remove from neighbors
        for neighbor in self.graph[idx].keys():
            self.graph[neighbor].pop(idx)
        # remove neighbors
        self.graph[idx] = dict()
        # set new point
        self.points[idx,:] = P
        # find corresponding triangle to get edges
        for triangle in self.triangles:
            a = triangle[0]
            b = triangle[1]
            c = triangle[2]

            A = self.points[a, :]
            B = self.points[b, :]
            C = self.points[c, :]
            if self.point_in_triangle(P, A, B, C):
                break
        # add distances to triangle
        self.graph[idx][a] = math.hypot(P[0]-A[0], P[1]-A[1]) / 10
        self.graph[idx][b] = math.hypot(P[0]-B[0], P[1]-B[1]) / 10
        self.graph[idx][c] = math.hypot(P[0]-C[0], P[1]-C[1]) / 10

        self.graph[a][idx] = self.graph[idx][a]
        self.graph[b][idx] = self.graph[idx][b]
        self.graph[c][idx] = self.graph[idx][c]

    def set_start(self, P):
        self.set_point(self.start_idx, P)

    def set_end(self, P):
        self.set_point(self.end_idx, P)

    def trace(self):
        return self.astar(self.start_idx, self.end_idx)

#endregion

#region Curriculum Generator

class Curriculum():
    def __init__(self, canvas_size, canvas_bounds):
        # experimentally recovered min-max, for more range later we can rescale the simulation
        self.thickness_bounds = np.array([0.12, 0.39])
        self.thickness = 0
        self.angle = None
        # based on camera distance of 5.5 and field of view of 120 degrees
        self.canvas_bounds = canvas_bounds
        self.canvas_size = canvas_size
        # prepare curriculum - linearly increase angel difficulty and exponential deviation
        self.distributions = np.zeros((10,2))
        for i in range(10):
            self.distributions[i,:] = [0, i/10]
        self.difficulty = 8
        self.epsilon = 1e-5
        self.outline_offset = 0.0138
        self.inline_offset = 0.034
        self.infill_offset = 0.02015
        self.join_threshold = 0.6
        self.slice_scale = 0.45

    def filterTrace(self, trace, segment_ids):
        filtered = [trace[0]]
        np_segment_ids = np.array(segment_ids)
        # remove duplicite vertices
        for idx in range(1, trace.shape[0]):
            if math.hypot(filtered[-1][0]-trace[idx,0],filtered[-1][1]-trace[idx,1]) > self.epsilon:
                filtered.append(trace[idx])
            else:
                for i in range(len(segment_ids)):
                    if segment_ids[i][0] >= idx:
                        np_segment_ids[i,0] -= 1
                    if segment_ids[i][1] >= idx:
                        np_segment_ids[i,1] -= 1
        return np.array(filtered), np_segment_ids

    def filterInfillSegments(self, locations, segment_ids, segment_types):
        # remove too short segments that would typically not require to lift the nozzle
        segment_ids_filtered = []
        segment_types_filtered = []
        for idx in range(segment_ids.shape[0]):
            dist = 0
            for i in range(segment_ids[idx,0]-1, segment_ids[idx,1]):
                P = locations[i, :]
                Q = locations[i+1, :]
                dist += math.sqrt((P[0]-Q[0])**2 + (P[1]-Q[1])**2)
            if (dist > self.join_threshold) or (idx == 0):
                segment_ids_filtered.append((segment_ids[idx,0], segment_ids[idx,1]))
                segment_types_filtered.append(segment_types[idx])
        return np.array(segment_ids_filtered), segment_types_filtered

    def filterSegments(self, locations, segment_ids, segment_types):
        # remove too short segments that would typically not require to lift the nozzle
        segment_ids_filtered = []
        segment_types_filtered = []
        for idx in range(segment_ids.shape[0]):
            dist = 0
            for i in range(segment_ids[idx,0]-1, segment_ids[idx,1]):
                P = locations[i, :]
                Q = locations[i+1, :]
                dist += math.sqrt((P[0]-Q[0])**2 + (P[1]-Q[1])**2)
            if (segment_types[idx] == True) or (idx > 0 and segment_types[idx-1] == True) or (dist > self.join_threshold) or (idx == 0):
                segment_ids_filtered.append((segment_ids[idx,0], segment_ids[idx,1]))
                segment_types_filtered.append(segment_types[idx])
        return np.array(segment_ids_filtered), segment_types_filtered

    def locations2global(self, locations):
        return locations * (self.canvas_bounds[1]-self.canvas_bounds[0]) + self.canvas_bounds[0]

    def dirs(self, locations):
        dirs = []
        for i in range(len(locations)-1):
            A = locations[i]
            B = locations[i+1]
            dir = B - A
            dir = dir / np.sqrt(dir.dot(dir))
            dirs.append(dir)
        dirs.append(dirs[-1])
        return np.array(dirs)

    def location2image(self, locations, thickness, height):
        img = Image.new('L', (self.canvas_size, self.canvas_size), 0)  
        draw = ImageDraw.Draw(img)
        draw.line(locations, fill=1, width=thickness, joint="curve")
        img = np.asarray(img) * height
        return img

    def locations2path(self, locations):
        locations[:,1] = 1.0 - locations[:,1]
        locations = (locations * self.canvas_size).flatten().tolist()
        return self.location2image(locations, 2, 1)

    def locations2targetpath(self, locations):
        locations[:,1] = 1.0 - locations[:,1]
        locations = (locations * self.canvas_size).flatten().tolist()
        if (self.thickness < 1):
            thickness = random.randint(14, 26)
        else:
            thickness = self.thickness
        alpha = random.random()
        height = 1
        return self.location2image(locations, thickness, height), self.location2image(locations, 3, 1)
    
    def locations2mask(self, locations):
        locations[:,1] = 1.0 - locations[:,1]
        locations = (locations * self.canvas_size).flatten().tolist()
        if (self.thickness < 1):
            thickness = random.randint(14, 26)
        else:
            thickness = self.thickness
        return self.location2image(locations, thickness, 1)

    def locations2cmask(self, locations):
        locations[:,1] = 1.0 - locations[:,1]
        locations = (locations * self.canvas_size).flatten().tolist()
        if (self.thickness < 1):
            thickness = random.randint(14, 26)
        else:
            thickness = self.thickness
        return self.locations2weights(locations, thickness)

    def locations2weights(self, locations, thickness):
        num_points = int(len(locations) / 2)
        weights = np.zeros((self.canvas_size, self.canvas_size))

        for i in range(num_points-1):
            idx = 2*i
            if i == 0:
                A = (locations[2*(num_points-2)+0], locations[2*(num_points-2)+1])
            else:
                A = (locations[idx-2], locations[idx-1])
            B = (locations[idx+0], locations[idx+1])
            C = (locations[idx+2], locations[idx+3])
            mask_locations = [
                0.2*A[0]+0.8*B[0], 0.2*A[1]+0.8*B[1],
                B[0], B[1],
                0.8*B[0]+0.2*C[0], 0.8*B[1]+0.2*C[1]
            ]
            weights += self.location2image(mask_locations, thickness, 1)

        np.clip(weights, 0, 1.0, out=weights)

        return weights

    def filter_fun(self, idx1, idx2, hole_ids):
        clipper_args = [pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD]
        pci = pyclipper.Pyclipper()
        pci.AddPath(hole_ids[idx1], pyclipper.PT_CLIP)
        pci.AddPath(hole_ids[idx2], pyclipper.PT_SUBJECT)
        solution = pci.Execute(*clipper_args)
        return len(solution) == 0

    def discretize(self, lines, delta, start_x = 0.0, end_x = 1.0, even_odd=False):
        cx = start_x
        idx = 0
        N = len(lines)
        active = []
        points = []
        active_points = []
        num_active = 0
        even = True
        while cx < end_x:
            # update active edge list
            for i in range(idx, N):
                if lines[i, 0] - self.epsilon <= cx:
                    active.append(lines[i,:])
                    idx = idx+1
                else:
                    break
            active = [x for x in active if x[1]-self.epsilon > cx]
            # add points if number of intersection changes
            if len(active) != num_active:
                if active_points:
                    points.append(active_points)
                active_points = []
                num_active = len(active)
            if active:
                # set intersection coordinate of each active edge
                intersections = [x[2] + x[4]*(cx-x[0]) for x in active]
                # sort intersections by y coordinate
                intersections.sort()
                # discretize points inside pairs of intersections
                M = int(len(intersections) / 2)
                if not active_points:
                    active_points = [[] for x in range(M)]
                for j in range(M):
                    if not even_odd or even:
                        active_points[j].append((cx, intersections[j*2 + 0], intersections[j*2 + 1]))
                    else:
                        active_points[j].append((cx, intersections[j*2 + 1], intersections[j*2 + 0]))
            cx += delta
            even = not even
        if active_points:
            points.append(active_points)
        return points

    def reroute(self, P, Q, navmesh):
        navmesh.set_start(P)
        navmesh.set_end(Q)
        foundPath = navmesh.trace()
        viapoints = np.array([0,0])
        for idx in foundPath:
            viapoints = np.vstack((viapoints, navmesh.points[idx]))
        return viapoints[2:-2,:]

    def locations2segments(self, locations):
        N = len(locations)-1
        lines = []
        for segment in locations:
            N = len(segment)-1
            for i in range(N):
                A = segment[i,:]
                B = segment[i+1,:]
                if A[0] > B[0]:
                    A, B = B, A
                if abs(B[0] - A[0]) < self.epsilon:
                    continue
                lines.append((A[0], B[0], A[1], B[1], (B[1] - A[1]) / (B[0] - A[0])))
        # order lines by starting and ending x
        lines.sort(key=lambda element: (element[0], element[1]))
        lines = np.array(lines)
        return lines

    def renderMesh(self, lines):
        delta = (1.0 / self.canvas_size)
        start = 0.5*delta
        segments = self.discretize(lines, delta, start)
        target = np.zeros((self.canvas_size, self.canvas_size))
        for row in segments:
            for chunk in row:
                for segment in chunk:
                    x_index = math.floor(segment[0] / delta)
                    # start and end are flipped due to the coordinate flip
                    y_end = self.canvas_size-math.floor(segment[1] / delta)
                    y_start = self.canvas_size-math.floor(segment[2] / delta)
                    target[y_start:y_end+1, x_index] = 1.0 # +1 to account for slicing that ends one before
        return target

    def traceInfillMesh(self, P, locations, lines_infill, navmesh, delta=0.01):
        infill_segments = self.discretize(lines_infill, delta, even_odd=True)

        segment_ids = []
        segment_types = []
        points = np.array(P)
        # linearize segments
        linear_segments = []
        for segment in infill_segments:
            for chunk in segment:
                linear_part = []
                for point in chunk:
                    linear_part.append([point[0], point[1]])
                    linear_part.append([point[0], point[2]])
                linear_segments.append(linear_part)
        # add segments to path
        ids = set(range(len(linear_segments)))
        last_outline = True
        while ids:
            # find closest segment to P
            min_dist = 10 # large enough for [0,1]
            min_idx = 0
            flip = False
            for idx in ids:
                dist1 = math.hypot(P[0]-linear_segments[idx][0][0], P[1]-linear_segments[idx][0][1])
                dist2 = math.hypot(P[0]-linear_segments[idx][-1][0], P[1]-linear_segments[idx][-1][1])
                if dist1 < min_dist:
                    min_dist = dist1
                    Q = np.array(linear_segments[idx][0])
                    min_idx = idx
                    flip = False
                if dist2 < min_dist:
                    min_dist = dist1
                    Q = np.array(linear_segments[idx][-1])
                    min_idx = idx
                    flip = True
            ids.remove(min_idx)
            # add segment to path
            viapoints = self.reroute(P, Q, navmesh)
            if last_outline or viapoints.shape[0] > 0:
                reroute_start_idx = points.shape[0]
                points = np.vstack((points, viapoints))
                reroute_end_idx = points.shape[0]
                segment_ids.append((reroute_start_idx,reroute_end_idx))
                segment_types.append(False)
            if flip:
                points = np.vstack((points, np.flipud(np.array(linear_segments[min_idx]))))
            else:
                points = np.vstack((points, np.array(linear_segments[min_idx])))
            # update P
            P = points[-1]
            last_outline = False
        return points, segment_ids, segment_types

    def traceMesh(self, locations, lines_infill, navmesh, delta=0.01):
        # start by adding all outline points
        segment_ids = []
        segment_types = []
        points = np.array(locations[0])
        P = points[-1]
        ids = set(range(1,len(locations)))
        while ids:
            # find closest node to P
            min_dist = 10 # large enough for [0,1]
            min_idx = 0
            for idx in ids:
                dist = math.hypot(P[0]-locations[idx][0][0], P[1]-locations[idx][0][1])
                if dist < min_dist:
                    Q = locations[idx][0]
                    min_dist = dist
                    min_idx = idx
            ids.remove(min_idx)
            # add to path
            viapoints = self.reroute(P, Q, navmesh)
            reroute_start_idx = points.shape[0]
            if viapoints.shape[0] > 0:
                points = np.vstack((points, viapoints))
            reroute_end_idx = points.shape[0]
            points = np.vstack((points, np.array(locations[min_idx])))
            segment_ids.append((reroute_start_idx,reroute_end_idx))
            segment_types.append(True)
            # pick new P
            P = points[-1]
        return points, segment_ids, segment_types

    def load3Dmesh(self, fname, plane_orig):
        # load mesh
        mesh = trimesh.load('data/stls/'+fname, process=True)
        # rescale mesh
        scaling = np.max(mesh.vertices[:,2]) - np.min(mesh.vertices[:,2])
        mesh.vertices -= 0.5*(np.max(mesh.vertices, axis=0) + np.min(mesh.vertices, axis=0))
        mesh.vertices /= scaling
        mesh.vertices += 0.5
        # cut mesh
        mesh = meshcut.TriangleMesh(mesh.vertices, mesh.faces)
        plane_norm = (0, 0, 1)
        plane = meshcut.Plane(plane_orig, plane_norm)
        P = meshcut.cross_section_mesh(mesh, plane)
        # get segment orientations
        orientations = dict()
        for i in range(len(P)):
            segment = P[i]
            area = 0
            ps = len(segment)
            for j in range(ps):
                next = j+1 * (j < ps-1)
                area += (segment[next,0]-segment[j,0])*(segment[next,1]+segment[j,1])
            clockwise = area > 0
            orientations[i] = clockwise
        # find largest segment
        largest_idx = -1
        largest_diag = 0
        for idx in range(len(P)):
            min_p = np.min(P[idx], axis=0)
            max_p = np.max(P[idx], axis=0)
            diag = np.sum((max_p-min_p)**2)
            if diag > largest_diag:
                largest_diag = diag
                largest_idx = idx

        largest_poly = [(X[0], X[1]) for X in P[largest_idx]]
        clipper_args = [pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD]
        # identify holes
        hole_ids = dict()
        for idx in range(len(P)):
            if idx == largest_idx:
                continue
            poly = [(X[0], X[1]) for X in P[idx]]
            poly = pyclipper.scale_to_clipper(poly)
            pci = pyclipper.Pyclipper()
            pci.AddPath(pyclipper.scale_to_clipper(largest_poly), pyclipper.PT_CLIP)
            pci.AddPath(poly, pyclipper.PT_SUBJECT)
            solution = pci.Execute(*clipper_args)
            if solution:
                hole_ids[idx] = poly

        filtered = []
        for idx in hole_ids.keys():
            if all(self.filter_fun(idx, other, hole_ids) for other in filtered):
                filtered.append(idx)
        hole_ids = filtered
        # rescale largest segment to view
        scaling = np.max(np.max(P[largest_idx], axis=0) - np.min(P[largest_idx], axis=0))
        offset = 0.5*(np.max(P[largest_idx], axis=0) + np.min(P[largest_idx], axis=0))
        for i in range(len(P)):
            P[i] = ((P[i]-offset) / scaling)*self.slice_scale
            P[i] = P[i] + 0.5
        # generate offset
        poly = []
        if orientations[largest_idx]:
            poly.append([(X[0], X[1]) for X in P[largest_idx]])
        else:
            poly.append([(X[0], X[1]) for X in reversed(P[largest_idx])])
        for idx in hole_ids:
            if orientations[idx]:
                poly.append([(X[0], X[1]) for X in reversed(P[idx])])
            else:
                poly.append([(X[0], X[1]) for X in (P[idx])])
        # negative offset for infill
        pco = pyclipper.PyclipperOffset()
        pco.AddPaths(pyclipper.scale_to_clipper(poly), pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
        solution = pco.Execute(pyclipper.scale_to_clipper(-self.inline_offset))
        solution = pyclipper.scale_from_clipper(solution)
        # positive offset for target
        render_offset = pco.Execute(pyclipper.scale_to_clipper(-self.outline_offset))
        render_offset = pyclipper.scale_from_clipper(render_offset)
        PP = P
        P = render_offset
        # find largest id and holes in offset polygon
        largest_idx_PP = largest_idx
        hole_ids_PP = hole_ids
        # find largest segment
        largest_idx = -1
        largest_diag = 0
        for idx in range(len(P)):
            min_p = np.min(P[idx], axis=0)
            max_p = np.max(P[idx], axis=0)
            diag = np.sum((max_p-min_p)**2)
            if diag > largest_diag:
                largest_diag = diag
                largest_idx = idx
        # identify holes
        hole_ids = dict()
        for idx in range(len(P)):
            if idx == largest_idx:
                continue
            poly = [(X[0], X[1]) for X in P[idx]]
            poly = pyclipper.scale_to_clipper(poly)
            hole_ids[idx] = poly

        filtered = []
        for idx in hole_ids.keys():
            if all(self.filter_fun(idx, other, hole_ids) for other in filtered):
                filtered.append(idx)
        hole_ids = filtered
        # triangulate navmesh
        pts = []
        seg = []
        hls = []
        
        seg_idx = len(P[largest_idx])
        pts = [(X[0], X[1]) for X in P[largest_idx]]
        seg = [(i, i+1) for i in range(seg_idx)]
        seg[-1] = (seg[-1][0], 0)

        for i in hole_ids:
            segment = P[i]

            local_pts = []
            local_seg = []

            for idx in range(len(segment)):
                X = segment[idx]
                pts.append((X[0], X[1]))
                local_pts.append((X[0], X[1]))
                if idx == len(segment)-1:
                    seg.append((seg_idx+idx, seg_idx))
                    local_seg.append((idx, 0))
                else:
                    seg.append((seg_idx+idx, seg_idx+idx+1))
                    local_seg.append((idx, idx+1))

            local_pts = np.array(local_pts)
            local_seg = np.array(local_seg, dtype=int)
            local_A = dict(vertices=local_pts, segments=local_seg)
            local_B = tr.triangulate(local_A, 'qp')
            local_pts = local_B['vertices']
            local_triangles = local_B['triangles']

            for T in local_triangles:
                if math.hypot(local_pts[T[0],0]-local_pts[T[1],0], local_pts[T[0],1] - local_pts[T[1],1]) > 0.001:
                    C = (local_pts[T[0],:] + local_pts[T[1],:] + local_pts[T[2],:]) / 3
                    break
            hls.append(C)

            seg_idx += idx + 1

        num_og_pts = len(pts)

        for segment in solution:
            for point in segment:
                pts.append((point[0], point[1]))

        pts = np.array(pts)
        seg = np.array(seg, dtype=int)
        hls = np.array(hls)

        if hls.size > 0:
            A = dict(vertices=pts, segments=seg, holes=hls)
        else:
            A = dict(vertices=pts, segments=seg)
        B = tr.triangulate(A, 'p')
        # construct graph for pathfinding
        points = B['vertices']
        triangles = B['triangles']
        graph = dict()
        for i in range(points.shape[0]):
            graph[i] = dict()
        for T in triangles:
            a = points[T[0],:]
            b = points[T[1],:]
            c = points[T[2],:]
            if T[0] < num_og_pts and T[1] < num_og_pts:
                graph[T[0]][T[1]] = math.hypot(a[0]-b[0], a[1]-b[1])
            else:
                graph[T[0]][T[1]] = math.hypot(a[0]-b[0], a[1]-b[1]) / 10
            if T[0] < num_og_pts and T[2] < num_og_pts:
                graph[T[0]][T[2]] = math.hypot(a[0]-c[0], a[1]-c[1])
            else:
                graph[T[0]][T[2]] = math.hypot(a[0]-c[0], a[1]-c[1]) / 10
            if T[1] < num_og_pts and T[2] < num_og_pts:
                graph[T[1]][T[2]] = math.hypot(b[0]-c[0], b[1]-c[1])
            else:
                graph[T[1]][T[2]] = math.hypot(b[0]-c[0], b[1]-c[1]) / 10

            graph[T[1]][T[0]] = graph[T[0]][T[1]]
            graph[T[2]][T[0]] = graph[T[0]][T[2]]
            graph[T[2]][T[1]] = graph[T[1]][T[2]]
        # return border locations and navmesh
        locations = []
        locations.append([(X[0], X[1]) for X in PP[largest_idx_PP]])
        locations[-1].append(locations[-1][0])
        locations[-1] = np.array(locations[-1])
        for idx in hole_ids_PP:
            locations.append([(X[0], X[1]) for X in PP[idx]])
            locations[-1].append(locations[-1][0])
            locations[-1] = np.array(locations[-1])
        infill = []
        for polygon in solution:
            infill.append(polygon)
            infill[-1].append(infill[-1][0])
            infill[-1] = np.array(infill[-1])
        outline = []
        for idx, polygon in enumerate(render_offset):
            if idx > 0:
                if not self.polygonOrientation(polygon):
                    tmp = polygon[0]
                    polygon = polygon[:0:-1]
                    polygon.insert(0, tmp)
            outline.append(polygon)
            outline[-1].append(outline[-1][0])
            outline[-1] = np.array(outline[-1])
        navmesh = PathFinder(points, graph, triangles)

        return outline, locations, infill, navmesh

    def polygonOrientation(self, poly):
        ids = list(range(len(poly)))
        ids.append(-1)

        clockwise = 0.
        for i in range(len(ids)-1):
            A = poly[ids[i]]
            B = poly[ids[i+1]]
            clockwise += (B[0]-A[0])*(B[1]+A[1])

        return clockwise > 0

    def estimatePathLength(self, global_locations, segment_ids):
        P = global_locations[0,:]
        i = 1
        j = 0
        distance = 0.
        while True:
            X = global_locations[i,:]
            distance += np.sqrt(np.sum(np.square(P-X)))
            P = X
            i += 1
            if segment_ids is not None:
                if j < segment_ids.shape[0]:
                    if i == segment_ids[j,0]-1:
                        i = segment_ids[j,1]
                        j += 1
            if i >= global_locations.shape[0]:
                break
        return distance

    def generate(self, mask_weights = [1, 1, 1], fixed_params = None, do_infill = False):
        if fixed_params is not None and type(fixed_params) is list:
            if len(fixed_params) == 3:
                return self.generateThingy(fixed_params[0], fixed_params[1], fixed_params[2])
        database = [
            (1, 0.75, 0.6),
            (2, 0.5, 0.6),
            (3, 0.5, 0.5),
            (4, 0.1, 0.5),
            (5, 0.5, 0.6),
            (6, 0.4, 0.5),
            (7, 0.5, 0.6),
            (8, 0.2, 0.4),
            (9, 0.95, 0.45),
            (10, 0.5, 0.5),
            (11, 0.5, 0.45),
            (12, 0.05, 0.8),
            (13, 0.5, 0.5),
            (14, 0.5, 0.5),
            (15, 0.5, 0.5),
            (16, 0.05, 0.5),
            (17, 0.05, 0.5),
            (18, 0.5, 0.55),
            (19, 0.5, 0.5),
            (20, 0.75, 0.4),
            (21, 0.05, 0.62),
            (22, 0.04, 0.55),
            (23, 0.75, 0.75),
            (24, 0.5, 0.75),
            (25, 0.5, 0.75),
            (26, 0.5, 0.75),
            (27, 0.5, 0.75),
            (28, 0.5, 0.8),
            (29, 0.5, 0.8),
            (27, 0.25, 0.75),
            (27, 0.75, 0.75),
            (27, 0.33333, 0.75),
            (30, 0.75, 0.75),
            (30, 0.5, 0.75),
            (30, 0.25, 0.75),
            (31, 0.5, 0.75),
            (32, 0.5, 0.75),
            (33, 0.5, 0.8),
            (34, 0.5, 0.75),
            (35, 0.5, 0.75),
            (36, 0.5, 0.8),
            (37, 0.5, 0.75),
            (38, 0.5, 0.75),
            (38, 0.25, 0.75),
            (38, 0.75, 0.75),
            (39, 0.5, 0.75),
            (39, 0.25, 0.75),
            (39, 0.75, 0.75),
            (40, 0.5, 0.75),
            (40, 0.05, 0.75),
            (41, 0.5, 0.75),
            (42, 0.5, 0.8),
            (43, 0.5, 0.75),
            (43, 0.25, 0.75),
            (44, 0.5, 0.75),
            (45, 0.5, 0.75),
            (46, 0.5, 0.75),
            (47, 0.5, 0.75),
            (48, 0.25, 0.75),
            (49, 0.5, 0.75),
            (50, 0.5, 0.75),
            (51, 0.5, 0.75),
            (51, 0.75, 0.75),
            (52, 0.5, 0.75),
            (53, 0.5, 0.75),
            (53, 0.75, 0.75),
            (54, 0.25, 0.75),
            (55, 0.5, 0.75),
            (56, 0.5, 0.75),
            (57, 0.5, 0.75),
            (58, 0.5, 0.75),
            (59, 0.5, 0.75),
            (60, 0.25, 0.75),
            (61, 0.05, 0.75),
            (62, 0.25, 0.75),
            (62, 0.05, 0.75),
            (63, 0.5, 0.75),
            (64, 0.5, 0.8),
            (65, 0.5, 0.8),
            (65, 0.28, 0.8),
            (66, 1.0, 0.8),
            (67, 0.5, 0.8),
            (67, 0.75, 0.8),
            (68, 0.25, 0.8),
            (69, 1.0, 0.8),
            (69, 0.25, 0.8),
            (70, 0.5, 0.8),
            (71, 0.75, 0.8),
            (72, 0.05, 0.8),
            (73, 0.05, 0.75),
            (74, 0.95, 0.8),
            (75, 0.5, 0.8),
            (76, 0.5, 0.75),
            (77, 0.5, 0.75),
            (78, 0.5, 0.8),
            (79, 0.5, 0.75),
            (80, 0.5, 0.8),
            (80, 0.75, 0.8),
            (81, 0.5, 0.8),
            (82, 0.5, 0.8),
            (83, 0.95, 0.8),
            (84, 0.95, 0.8),
            (85, 0.5, 0.8),
            (86, 0.5, 0.8),
            (87, 0.5, 0.8),
            (88, 0.95, 0.8),
            (89, 0.5, 0.8),
            (90, 0.25, 0.8),
            (91, 0.5, 0.8),
            (92, 0.75, 0.8),
            (93, 0.95, 0.75),
            (94, 0.95, 0.8),
            (95, 0.95, 0.8),
            (96, 0.95, 0.8),
            (97, 0.95, 0.8),
            (98, 0.5, 0.8),
            (99, 0.5, 0.75),
            (100, 0.5, 0.75),
            (101, 0.05, 0.75),
            (102, 0.5, 0.8),
            (103, 0.05, 0.8),
            (104, 0.5, 0.75),
            (105, 0.5, 0.8),
            (106, 0.95, 0.8),
            (107, 0.5, 0.8),
            (108, 0.5, 0.8),
            (109, 0.95, 0.75),
            (109, 0.5, 0.8),
            (110, 0.5, 0.8),
            (111, 0.5, 0.8),
            (112, 0.5, 0.75),
            (113, 0.5, 0.75),
            (114, 0.5, 0.8),
            (115, 0.5, 0.755),
            (116, 0.5, 0.75),
            (117, 0.5, 0.84),
        ]
        select = random.randint(0, len(database)-1)
        if not (fixed_params is None):
            select = fixed_params
        mesh_idx = database[select][0]
        mesh_cut_height = database[select][1]
        self.slice_scale = database[select][2]
        locations, outline, infill, navmesh = self.load3Dmesh(str(mesh_idx)+'.stl', (0,0,mesh_cut_height))
        lines_outline = self.locations2segments(outline)
        lines_infill = self.locations2segments(infill)

        trace, segment_ids, segment_types = self.traceMesh(locations, lines_infill, navmesh, delta=self.infill_offset)
        if do_infill is True:
            trace_infill, segment_ids_infill, segment_types_infill = self.traceInfillMesh(trace[-1], locations, lines_infill, navmesh, delta=self.infill_offset)

        trace, segment_ids = self.filterTrace(trace, segment_ids)
        global_locations = self.locations2global(trace)
        segment_ids, segment_types = self.filterSegments(global_locations, segment_ids, segment_types)
        directions = self.dirs(global_locations)

        if do_infill is True:
            trace_infill, segment_ids_infill = self.filterTrace(trace_infill, segment_ids_infill)
            global_locations_infill = self.locations2global(trace_infill)
            segment_ids_infill, segment_types_infill = self.filterInfillSegments(global_locations_infill, segment_ids_infill, segment_types_infill)
            directions_infill = self.dirs(global_locations_infill)

        target = self.renderMesh(lines_outline)
        weights = np.array(target)
        path = self.locations2path(trace)
        if do_infill is True:
            path = self.locations2path(trace_infill)

        # path length
        distance = self.estimatePathLength(global_locations, segment_ids)

        if do_infill is True:
            return global_locations, directions, segment_ids, segment_types, target, path, weights, distance, global_locations_infill, directions_infill, segment_ids_infill
        
        return global_locations, directions, segment_ids, segment_types, target, path, weights, distance

    def generateThingy(self, mesh_name, mesh_cut_height, mesh_slice_scale):
        self.slice_scale = mesh_slice_scale
        
        locations, outline, infill, navmesh = self.load3Dmesh(mesh_name, (0,0,mesh_cut_height))
        lines_outline = self.locations2segments(outline)
        lines_infill = self.locations2segments(infill)

        trace, segment_ids, segment_types = self.traceMesh(locations, lines_infill, navmesh, delta=self.infill_offset)

        trace, segment_ids = self.filterTrace(trace, segment_ids)
        global_locations = self.locations2global(trace)
        segment_ids, segment_types = self.filterSegments(global_locations, segment_ids, segment_types)
        directions = self.dirs(global_locations)

        target = self.renderMesh(lines_outline)
        weights = np.array(target)
        path = self.locations2path(trace)

        # path length
        distance = self.estimatePathLength(global_locations, segment_ids)

        return global_locations, directions, segment_ids, segment_types, target, path, weights, distance
