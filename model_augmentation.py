import numpy as np
from scipy.spatial.transform import Rotation


class ModelAugmenter:
    def rotate(self, mesh, angle=45.0):
        """Rotate model around Y axis"""
        # Create rotation matrix around Y axis
        rotation = Rotation.from_euler('y', angle, degrees=True)
        # Apply rotation to vertices
        mesh.vertices = rotation.apply(mesh.vertices)
        return mesh

    def scale_random(self, mesh, min_scale=0.8, max_scale=1.2):
        """Random non-uniform scaling"""
        scale_factors = np.random.uniform(min_scale, max_scale, size=3)
        mesh.vertices *= scale_factors
        return mesh

