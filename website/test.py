import numpy as np
import os 
import open3d as o3d
from utils import get_bounding_boxes
from utils import COLOR_DETECTRON2
import random
import numpy as np


def generate_random_point_in_sphere(radius):
  """Generates a random point inside a sphere with the given radius.

  Args:
    radius: The radius of the sphere.

  Returns:
    A tuple of three floats representing the (x, y, z) coordinates of the point.
  """

  # Generate three random numbers between -1 and 1.
  x = random.uniform(-1, 1)
  y = random.uniform(-1, 1)
  z = random.uniform(-1, 1)

  # Normalize the vector so that it has a magnitude of 1.
  norm = np.linalg.norm([x, y, z])
  x /= norm
  y /= norm
  z /= norm

  # Scale the vector by the radius of the sphere.
  x *= radius
  y *= radius
  z *= radius

  return x, y, z


from tqdm import tqdm
if __name__=='__main__':
    get_bounding_boxes('/home/manas/test/3DOD/website/static/output/scene0000_00_vh_clean_2')