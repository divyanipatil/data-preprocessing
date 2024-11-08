import trimesh
import numpy as np
import io
import base64
import json


class ModelPreprocessor:
    def load_obj(self, obj_data):
        """Load OBJ file from bytes"""
        try:
            obj_file = io.BytesIO(obj_data)
            mesh = trimesh.load(
                obj_file,
                file_type='obj',
                force='mesh'
            )

            if isinstance(mesh, trimesh.Scene):
                meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                if not meshes:
                    raise ValueError("No valid meshes found in scene")
                mesh = meshes[0]

            print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
            return mesh

        except Exception as e:
            print(f"Load error: {str(e)}")
            raise ValueError(f"Failed to load mesh: {str(e)}")

    def normalize(self, mesh):
        """Normalize vertices to fit in a unit sphere"""
        try:
            normalized = mesh.copy()

            # Get current size info
            vertices = normalized.vertices
            centroid = vertices.mean(axis=0)
            distances = np.linalg.norm(vertices - centroid, axis=1)
            current_max_radius = np.max(distances)

            print(f"Before normalization:")
            print(f"Max radius from center: {current_max_radius}")
            print(f"Model dimensions: {mesh.bounds[1] - mesh.bounds[0]}")

            # Scale vertices to fit in unit sphere
            scale_factor = 50.0 / current_max_radius
            normalized.vertices = centroid + (vertices - centroid) * scale_factor

            print(f"After normalization:")
            print(f"Scale factor applied: {scale_factor}")
            print(f"New max radius: {np.max(np.linalg.norm(normalized.vertices - centroid, axis=1))}")
            print(f"New dimensions: {normalized.bounds[1] - normalized.bounds[0]}")

            return normalized

        except Exception as e:
            print(f"Normalization error: {str(e)}")
            return mesh.copy()

    def center_model(self, mesh):
        """Center the model at origin"""
        try:
            centered = mesh.copy()
            centroid = mesh.centroid
            centered.vertices -= centroid

            print(f"Original centroid: {centroid}")
            print(f"After centering centroid: {centered.centroid}")
            return centered

        except Exception as e:
            print(f"Centering error: {str(e)}")
            return mesh.copy()

    def to_json(self, mesh):
        """Convert mesh to JSON format for three.js"""
        try:
            vertices = mesh.vertices.tolist()
            faces = mesh.faces.tolist()

            data = {
                'vertices': vertices,
                'faces': faces
            }

            return base64.b64encode(json.dumps(data).encode()).decode()

        except Exception as e:
            print(f"JSON conversion error: {str(e)}")
            raise ValueError(f"Failed to convert mesh to JSON: {str(e)}")
