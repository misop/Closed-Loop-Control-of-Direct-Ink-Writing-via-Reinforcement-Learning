import ctypes
from ctypes import *
import numpy as np
from numpy.ctypeslib import ndpointer
import datetime

class Pyflex:
    def __init__(self):
        self.lib = ctypes.CDLL('./pyflexlib.so')

        self.lib.CreatePyflexInstance.argtypes = []
        self.lib.CreatePyflexInstance.restype = ctypes.c_void_p

        self.lib.DeletePyflexInstance.argtypes = [ctypes.c_void_p]
        self.lib.DeletePyflexInstance.restype = ctypes.c_void_p

        self.lib.InitPyflexInstance.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int]
        self.lib.InitPyflexInstance.restype = ctypes.c_void_p

        self.lib.ResetPyflexInstance.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_int]
        self.lib.ResetPyflexInstance.restype = ctypes.c_void_p

        self.lib.StepPyflexInstance.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
        self.lib.StepPyflexInstance.restype = ctypes.c_void_p

        self.lib.TestPyflexInstance.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]
        self.lib.TestPyflexInstance.restype = ctypes.c_void_p

        self.obj = self.lib.CreatePyflexInstance()

    def __del__(self):
        self.lib.DeletePyflexInstance(self.obj)

    def init(self, width, height, border = 9, renderDevice = 0):
        self.lib.InitPyflexInstance(self.obj, width, height, border, -border, border, -border, renderDevice)
        self.width = width
        self.height = height

    def reset(self, numSubsteps, materialViscosity, materialReservoir):
        self.lib.ResetPyflexInstance(self.obj, numSubsteps, materialViscosity, materialReservoir)

    def step(self, x, y, z, flow):
        frame = np.zeros((self.width,self.height))
        self.lib.StepPyflexInstance(self.obj, x, y, z, flow, frame)
        return frame
    
    def test(self):
        frame = np.zeros((512,512))
        self.lib.TestPyflexInstance(self.obj, frame)
        frame = np.flipud(frame)
        return frame    

