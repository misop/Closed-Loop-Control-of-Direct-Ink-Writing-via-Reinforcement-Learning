import gym
from gym import spaces
import math
import numpy as np
from numpy import genfromtxt
import random
import queue
from PIL import Image
from PIL import ImageDraw
from matplotlib import pyplot as plt
import scipy
from scipy import ndimage
from skimage import transform as tf
from hilbert import decode
from environments.curriculum_slicer import Curriculum
import importlib
import sys
import cv2
import librosa

#region Moving Printer

flex_env_count = 0

class FlexPrinterEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        super(FlexPrinterEnv, self).__init__()
        # set screen size
        self.canvas_size = 512
        self.mioptic_size = 84 # indexing only implemented for even values
        self.canvas_bounds = np.array([-9, 9])
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.mioptic_size, self.mioptic_size, 3), dtype=np.uint8)
        self.pyflex = None
        self.pyflexModule = None
        self.obs = np.zeros((self.mioptic_size, self.mioptic_size, 3))
        # simulation timestep
        self.h = 1.0 / 60.0
        # helpers for fast jetting
        self.xs = np.zeros((self.mioptic_size,self.mioptic_size))
        self.ys = np.zeros((self.mioptic_size,self.mioptic_size))
        for i in range(self.mioptic_size):
            self.xs[i,:] = i / float(self.mioptic_size)
            self.ys[:,i] = i / float(self.mioptic_size)
        # plotting figure
        self.fig = None
        self.plt = 0
        # setup curriculum
        self.curriculum = Curriculum(self.canvas_size, self.canvas_bounds)
        # if fluid is higher we consider it to be inside the nozzle
        self.nozzle_threshold = 0.8
        self.nozzle_diameter = 0.8
        # step inputs
        self.delta_pos = 0.1
        self.height_limit = (0.75, 1.75)
        self.flow_limit = (0, 4)
        self.speed_limit = (0.2, 2.0)
        self.accel_scale = 3
        self.dist_delta = 0.266666666
        self.move_delta = 0.266666666

        self.pixel_size = (18.0/512.0)**2
        self.test_flag = 0
        self.sim_params = None
        self.fluid_params = [3, 1.0]
        self.gpu_id = 0

        self.prepare_pressure()

    def seed(self, seed):
        np.random.seed(seed)

    def setGPU(self, gpu_id):
        self.gpu_id = gpu_id

    def getObservationSpace(self):
        return spaces.Box(low=0, high=255, shape=(3, self.mioptic_size, self.mioptic_size), dtype=np.uint8)

    def get_observation(self):
        self.obs[:,:,0] = np.clip(self.mioptic_canvas * (1.0-self.mioptic_mask) * 2.0, 0.0, 0.9999) # scaling to increase the range
        self.obs[:,:,1] = self.mioptic_target
        self.obs[:,:,2] = self.mioptic_path

        return (self.obs*255).astype(np.uint8)

    def prepare_pressure(self):
        # measured deposition width
        filter_order = 3
        y = np.array([
            16, 17, 17, 15, 13, 12, 14, 17, 16, 15, 17, 21, 16, 15, 16, 14, 14, 14, 15, 15, 13, 14, 16, 15, 16, 15, 17, 15, 13, 15,
            20, 20, 20, 20, 16, 19, 18, 21, 19, 22, 18, 21, 21, 22, 20, 21, 20, 19, 20, 20, 20 ,18, 19, 18, 18, 20, 21, 18, 20, 21,
            19, 18, 19, 19, 17, 15, 17, 19, 18, 16, 18, 19, 16, 17, 19, 19, 18, 19, 17, 18, 18, 17, 19, 18, 16, 17, 19, 19, 18, 19,
            19, 18, 18, 20, 14, 15, 15, 15, 13, 14, 13, 14, 14, 13, 14, 13, 14, 14, 13, 13, 15, 16, 13, 14, 13, 16, 19, 18, 15, 14,
            16, 16, 17, 17, 11, 12, 13, 16, 15, 14, 14, 18, 17, 17, 17, 19, 22, 23, 23, 26, 24, 23, 22, 16, 13, 14, 18, 20, 24, 27,
            12, 12, 13, 12, 11, 11, 12, 15, 14, 14, 14, 16, 13, 13, 14, 13, 10, 12, 11, 12, 11, 10, 10, 15, 13, 12, 14, 12, 11, 13,
            11, 13, 13, 13, 11, 12, 12, 15, 13, 14, 13, 15, 13, 13, 13, 13, 10, 11, 10, 13,  9,  9, 11, 11, 12, 12, 13, 11, 11, 14,
            11, 12, 11, 11,  9, 10, 11, 12, 13, 14, 12, 12, 12, 12, 12, 14, 10, 10, 10, 11,  9,  9,  9, 10, 10,  9, 12, 11, 10, 11,
            11, 13, 13, 13, 12, 12, 15, 16, 18, 19, 19, 17, 15, 16, 15, 16, 12, 14, 13, 15, 13, 12, 12, 17, 21, 19, 17, 13, 16, 16,
        ])
        # 12 is the reference thickness
        y = y - 12
        # scaling to get material thickness to pressure
        y = y/3
        y = y - np.mean(y)
        a = librosa.lpc(y, filter_order)
        self.filter_b = np.hstack([[0], -1 * a[1:]])
        y_hat = scipy.signal.lfilter(self.filter_b, [1], y)
        self.filter_residual = y - y_hat

        self.make_pressure_samples()

    def make_pressure_samples(self):
        samples = np.random.normal(0, 0.15, size=270) + self.filter_residual
        y_p = scipy.signal.lfilter(self.filter_b, [1], samples)

        f = scipy.interpolate.UnivariateSpline(np.linspace(0, 30*9, 270), y_p)
        f.set_smoothing_factor(3)
        self.pressure_list = np.clip(f(np.linspace(0, 30*9, 1440))+0.5, -0.25, 2.0)
        self.pressure_list_idx = 0

    def get_pressure(self):
        if self.pressure_list_idx >= self.pressure_list.shape[0]:
            self.make_pressure_samples()

        pressure = self.pressure_list[self.pressure_list_idx]
        self.pressure_list_idx += 1
        return pressure

    def step(self, action, noisy_pressure=True):
        action = np.clip(action, -1., 1.)
        dx, dy = 0, 0
        a_height = 0
        v_xy = (action[0]+1.0)/2.0
        v_xy = (1-v_xy)*self.speed_limit[0] + v_xy*self.speed_limit[1]
        dv_xy = action[1]
        if noisy_pressure:
            flow = (-0.5 + 1.0)*0.50*self.flow_limit[1] + self.get_pressure()
        else:
            flow = (-0.5 + 1.0)*0.50*self.flow_limit[1]

        to_travel = self.dist_delta
        early_stop = False

        step_output = []
        while to_travel > 0:
            if self.location_idx >= self.locations.shape[0]:
                rendered_scene = self.pyflex.step(self.pos[0], self.height, self.pos[1], 0.0)
                step_output.append((self.pos[0], self.height, self.pos[1], 0.0))
                break
            # accelerate to desired speed
            a_xy = (v_xy - self.velocity)/self.h
            if a_xy < -1:
                a_xy = -1
            elif a_xy > 1:
                a_xy = 1
            self.velocity += a_xy * self.h
            if self.velocity > self.speed_limit[1]:
                self.velocity = self.speed_limit[1]
            if self.velocity < self.speed_limit[0]:
                self.velocity = self.speed_limit[0]
            # move to desired displacement
            da_xy = (dv_xy - self.d_xy) / self.h
            if da_xy < -2:
                da_xy = -2
            elif da_xy > 2:
                da_xy = 2
            self.d_xy += da_xy * self.h
            if self.d_xy > 1.0:
                self.d_xy = 1.0
            elif self.d_xy < -1.0:
                self.d_xy = -1.0
            # check distance we travel in one step
            distance = self.velocity * self.h
            to_travel -= distance
            skip = False
            Cx = self.pos[0]
            Cy = self.pos[1]
            # get new position based on the travel distance
            while True:
                direction = self.locations[self.location_idx] - self.pos
                dist = np.sqrt(direction.dot(direction))

                x0 = self.pos[0]
                y0 = self.pos[1]
                x1 = self.locations[self.location_idx, 0] + self.dir[1]*self.d_xy*self.move_delta
                y1 = self.locations[self.location_idx, 1] - self.dir[0]*self.d_xy*self.move_delta
                r = distance

                a = (x1 - x0)**2 + (y1 - y0)**2
                b = 2.0*(x1 - x0)*(x0 - Cx) + 2.0*(y1 - y0)*(y0 - Cy)
                c = (x0 - Cx)**2 + (y0 - Cy)**2 - r**2

                if a == 0:
                    t = 1000
                else:
                    t = (-b + math.sqrt(b**2 - 4.0*a*c)) / (2.0*a)
                if t > 0 and t <= 1.0:
                    self.pos[0] = x0 + t*(x1 - x0)
                    self.pos[1] = y0 + t*(y1 - y0)
                    break
                else:
                    self.location_idx += 1
                    if len(self.segments) > 0:
                        if self.location_idx == self.segments[self.segment_idx, 0]:#we reached a move to directive
                            # do one step
                            self.pos = self.locations[self.location_idx-1]
                            rendered_scene = self.pyflex.step(self.pos[0], self.height, self.pos[1], flow)
                            step_output.append((self.pos[0], self.height, self.pos[1], flow))
                            # move printing head up
                            hv = 0.0
                            done_move = False
                            while not done_move:
                                hv += 1.0 * self.h
                                if hv >= self.speed_limit[1]:
                                    hv = self.speed_limit[1]
                                self.height += hv * self.h
                                if self.height >= self.height_limit[1]:
                                    self.height = self.height_limit[1]
                                    done_move = True
                                rendered_scene = self.pyflex.step(self.pos[0], self.height, self.pos[1], 0.0)
                                step_output.append((self.pos[0], self.height, self.pos[1], 0.0))
                            # move to end segment
                            hv = 0.0
                            done_move = False
                            direction = self.locations[self.segments[self.segment_idx, 1]] - self.pos
                            dist = np.sqrt(direction.dot(direction))
                            if abs(dist) > 1e-3:
                                dist_moved = 0.0
                                while not done_move:
                                    hv += 1.0 * self.h
                                    if hv >= self.speed_limit[1]:
                                        hv = self.speed_limit[1]
                                    distance = hv*self.h
                                    dist_moved += distance
                                    if dist_moved >= dist:
                                        done_move = True
                                        distance = distance + dist - dist_moved
                                    self.pos += distance * direction/dist
                                    rendered_scene = self.pyflex.step(self.pos[0], self.height, self.pos[1], 0.0)
                                    step_output.append((self.pos[0], self.height, self.pos[1], 0.0))
                            # move printing head down
                            hv = 0.0
                            done_move = False
                            while not done_move:
                                hv += -1.0 * self.h
                                if -hv >= self.speed_limit[1]:
                                    hv = -self.speed_limit[1]
                                self.height += hv * self.h
                                if self.height <= self.height_limit[0]:
                                    self.height = self.height_limit[0]
                                    done_move = True
                                rendered_scene = self.pyflex.step(self.pos[0], self.height, self.pos[1], 0.0)
                                step_output.append((self.pos[0], self.height, self.pos[1], 0.0))
                            # update parameters
                            self.pixel_pos = self.position2pixel(self.pos)
                            self.location_idx = self.segments[self.segment_idx, 1] + 1
                            self.dir = self.dirs[self.location_idx-1,:]
                            self.velocity = 0.0
                            self.segment_idx = min(self.segment_idx+1, self.segments.shape[0]-1)
                            skip = True
                            break
                    if self.location_idx >= self.locations.shape[0]:# we are at the end
                        self.pos = self.locations[-1,:]
                        self.dir = self.dirs[-1,:]
                        break
                    else:#consume the distance and go next
                        self.dir = self.dirs[self.location_idx-1,:]
            if skip:
                break
            # simulate the movement
            self.pixel_pos = self.position2pixel(self.pos)
            rendered_scene = self.pyflex.step(self.pos[0], self.height, self.pos[1], flow)
            step_output.append((self.pos[0], self.height, self.pos[1], flow))

        self.canvas = np.array(rendered_scene).astype(float)
        canvas_binary = (rendered_scene > 0).astype(float)

        self.mioptic_canvas = self.make_mioptic(self.canvas)
        self.mioptic_target = self.make_mioptic(self.target)
        self.mioptic_path = self.make_mioptic(self.path)
        self.mioptic_mask = self.make_mask()
        # calculate reward
        # percent covered
        temp = canvas_binary * (1.0 - self.mask)
        percent_covered = np.sum(temp*self.target_mask*self.reward_weights) / self.max_reward
        # percent overflown
        percent_overflown = np.sum(temp*(1-self.target_mask)) / self.max_punishment
        # deviation of surface
        masked_values = np.sum(temp)
        if masked_values > 0:
            masked_mean = np.sum(rendered_scene * (1.0 - self.mask)) / masked_values
            masked_deviation = math.sqrt(np.sum(np.square((rendered_scene - masked_mean)*(self.canvas)*(1.0 - self.mask))) / masked_values)
            masked_deviation = masked_deviation / masked_mean
        else:
            masked_deviation = 0
        # total reward
        total_reward = percent_covered - percent_overflown - masked_deviation
        reward = total_reward - self.past_reward
        self.past_reward = total_reward
        # check if done
        done = True if self.location_idx >= self.locations.shape[0] or early_stop else False

        info = {
            'step_params': step_output
        }

        self.steps += 1

        return self.get_observation(), reward, done, info

    def increase_difficulty(self):
        self.curriculum.difficulty += 1

    def set_test_flag(self, flag):
        self.test_flag = flag

    def set_thickness(self, thickness):
        self.curriculum.thickness = thickness

    def set_angle(self, angle):
        self.curriculum.angle = angle

    def set_meshid(self, meshid):
        self.sim_params = meshid

    def reset(self):
        if self.pyflex is None:
            pyflex_package_name = 'environments.pyflex'
            self.pyflexModule = importlib.import_module(pyflex_package_name)
            self.pyflex = self.pyflexModule.Pyflex()
            self.pyflex.init(self.canvas_size, self.canvas_size, self.gpu_id)
        
        substeps = self.fluid_params[0]
        viscosity = self.fluid_params[1]
        material_reservoir = 1024*1024 # might need more for longer prints
        if self.test_flag > 0:
            material_reservoir = 1024*1024*2
        self.pyflex.reset(substeps, viscosity, material_reservoir)

        self.canvas = np.zeros((self.canvas_size, self.canvas_size))
        # generate random environment
        self.locations_outline, self.dirs_outline, self.segments_outline, self.segment_types, self.target, self.path, self.reward_weights, self.outline_length, self.locations_infill, self.dirs_infill, self.segments_infill = self.curriculum.generate([1, 1, 1], fixed_params=self.sim_params, do_infill=True)
        self.target_mask = (self.target > 0).astype(float)
        self.target_mask = self.target_mask / np.max(self.target_mask)
        self.reward_weights = np.array(self.target_mask)
        punish_mask = 1.0 - self.reward_weights
        # do the outline
        self.locations, self.dirs, self.segments = self.locations_outline, self.dirs_outline, self.segments_outline
        self.height = 0.75
        self.pos = self.locations[0,:]
        self.pixel_pos = self.position2pixel(self.pos)
        self.dir = self.dirs[0,:]

        self.mioptic_canvas = self.make_mioptic(self.canvas)
        self.mioptic_target = self.make_mioptic(self.target)
        self.mioptic_path = self.make_mioptic(self.path)
        self.mioptic_mask = self.make_mask()

        rendered_scene = self.pyflex.step(self.pos[0], self.height, self.pos[1], 0.0)  
        self.location_idx = 1
        self.segment_idx = 0
        self.velocity = 0
        self.d_xy = 0
        self.height_velocity = 0
        self.past_reward = 0
        self.steps = 0
        self.max_reward = np.sum(self.target_mask*self.reward_weights)
        self.max_punishment = self.max_reward
        outline_speed = random.random()
        outline_speed = (1-outline_speed)*0.2 + outline_speed*1.0
        outline_speed = 0.2

        done = False
        self.reset_step_params = []
        while not done:
            _, _, done, info = self.step(np.array([outline_speed, 0.0]), noisy_pressure=False)
            step_params = info['step_params']
            self.reset_step_params.extend(step_params)
        # prepare for infill
        self.locations, self.dirs, self.segments = self.locations_infill, self.dirs_infill, self.segments_infill
        # set the initial position of the printing head
        self.height = 0.75
        self.pos = self.locations[0,:]
        self.pixel_pos = self.position2pixel(self.pos)
        self.dir = self.dirs[0,:]
        # setup global mask
        self.mioptic_canvas = self.make_mioptic(self.canvas)
        self.mioptic_target = self.make_mioptic(self.target)
        self.mioptic_path = self.make_mioptic(self.path)
        self.mioptic_mask = self.make_mask()
		
        rendered_scene = self.pyflex.step(self.pos[0], self.height, self.pos[1], 0.0)
        
        # reward normalization
        height = np.max(self.target)
        factor = height if height >= self.nozzle_threshold/2 else (self.nozzle_threshold-height)

        self.location_idx = 1
        self.segment_idx = 0
        self.velocity = 0
        self.d_xy = 0
        self.height_velocity = 0
        self.timer = 0
		
        self.past_reward = 0
        self.avg_velocity = 0
        self.steps = 0

        return self.get_observation()  # reward, done, info can't be included

    def make_mask(self):
        self.mask = np.zeros((self.canvas_size, self.canvas_size))
        drop = self.jet(0.5, 0.5, self.nozzle_diameter)
        a = int(self.mioptic_size/2)
        self.mask[self.pixel_pos[0]-a:self.pixel_pos[0]+a, self.pixel_pos[1]-a:self.pixel_pos[1]+a] += drop

        return drop

    def get_angle(self, x, y):
        angle = np.rad2deg(math.atan2(y, x))
        if y < 0:
            angle += 360
        return angle

    def make_mioptic(self, M):
        angle = -self.get_angle(self.dir[0], self.dir[1])
        order = 0

        side = int(self.canvas_size / 2)
        shift = (side-self.pixel_pos[0], side-self.pixel_pos[1])
        shiftb = (-shift[0], -shift[1])
        
        tM = ndimage.shift(M, shift, order=order)
        tM = ndimage.rotate(tM, angle, reshape=False, order=order)
        tM = ndimage.shift(tM, shiftb, order=order)

        a = int(self.mioptic_size/2)
        return tM[self.pixel_pos[0]-a:self.pixel_pos[0]+a, self.pixel_pos[1]-a:self.pixel_pos[1]+a]

    def position2pixel(self, P):
        P = np.flipud(P)
        P = (((P + 9) / 18)*self.canvas_size).astype(int)
        P[0] = self.canvas_size - P[0]
        return P

    def clear_fig(self):
        self.fig = None

    def jet(self, x, y, r):
        droplet = np.zeros((self.mioptic_size, self.mioptic_size))
        r = (0.09523809523*r + 0.0238095238)**2

        droplet[(np.square(self.xs - x) + np.square(self.ys - y)) < r] = 1.0

        return droplet

    def render_preview(self, fname=None):
        if not self.fig:
            self.fig, self.plts = plt.subplots(1, 1)
        self.plts.clear()
        self.plts.tick_params(axis='both', which='both', bottom=False,top=False,labeltop=False,labelleft=False,labelbottom=False)
        I1 = np.zeros((512,512, 3))
        for i in range(3):
            I1[:,:,i] = self.target
        I1[:,:,0] -= (self.canvas)*self.target*(1.0-self.mask)
        I1[:,:,2] -= (self.canvas)*self.target*(1.0-self.mask)
        I1[:,:,0] += (self.canvas)*(1.0-self.target)*(1.0-self.mask)
        self.plts.imshow(I1)
        if not (fname is None):
            plt.savefig('tmp/'+fname+'.png')

        plt.show(block=False)
        plt.pause(0.001)


#endregion

