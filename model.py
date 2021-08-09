import numpy as np
from scipy.interpolate import splprep, splev
from matplotlib import pyplot as plt
from sys import argv
import glob
import xml.etree.ElementTree as ET
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R


FILTER_DISTANCE = 1000
INTERPOLATION_PERIOD = 4


class TomoModel():
    """Top level model class for Tomo data"""

    def __init__(self, name, pixel_size, model_range_tuple, offset_tuple):
        self.name = name
        self.number_of_objects = 0
        self.pixel_size = pixel_size
        self.model_range = model_range_tuple
        self.offset = offset_tuple
        self.objects = {}

    def add_object(self, newobject):
        self.number_of_objects += 1
        self.objects[newobject.name] = newobject

    def get_objects(self):
        return self.objects

    def get_object(self, name):
        return self.objects[name]


class ModelComponent():
    """Object class for data from Amira"""

    def __init__(self, name):
        self.name = name
        self.number_of_contours = 0
        self.contours = []
        self.datatype = ""

    def add_contour(self, newcontour):
        self.number_of_contours += 1
        self.contours.append(newcontour)

    def get_contours(self):
        return self.contours


class FilamentContour():
    """Contour class based on list of connected vertices"""

    def __init__(self, vertices, interpolate=True, interpolation_period_nm=INTERPOLATION_PERIOD, randomize=False, tomorange = None):
        vertices = np.array(vertices)
        self.original_vertices = vertices
        vectors = np.diff(vertices, axis=0)
        self.original_vectors = np.append(vectors,vectors[-1:],axis=0)
        assert len(self.original_vectors) == len(self.original_vertices)
        self.calculate_length_nm()
        if interpolate:
            self.interpolate_vertices(interpolation_period_nm, randomize=randomize, tomorange=tomorange)

    def calculate_length_nm(self):
        if len(self.original_vertices) < 2:
            self.length = 0
        else:
            length = 0
            for i in range(len(self.original_vertices)-1):
                length += np.linalg.norm(np.subtract(
                    self.original_vertices[i+1], self.original_vertices[i]))
            self.length = length
        return self.length

    def interpolate_vertices(self, interpolation_period_nm, randomize=False, tomorange=None):
        self.interpolation_period_nm = interpolation_period_nm
        if len(self.original_vertices) < 2:
            self.interpolated_vertices = self.original_vertices
        else:
            # TODO - FIGURE OUT HOW TO MAKE THIS NOT HAVE ROUNDING ERRORS
            x, y, z = zip(*self.original_vertices)
            if len(x) > 3:
                tck, u = splprep([x, y, z], k=1)
            else:
                tck, u = splprep([x, y, z], k=1)
            u_fine = np.linspace(
                u[0], u[-1], int(self.length/interpolation_period_nm))
            x_fine, y_fine, z_fine = splev(u_fine, tck)
            #v_x, v_y, v_z = splev(u_fine, tck, der=1)
            self.interpolated_vertices = np.array(
                [x_fine, y_fine, z_fine]).transpose()
            vectors = np.diff(self.interpolated_vertices, axis=0)
            # print(sum(np.linalg.norm(vectors, axis=1)), self.length)
            self.interpolated_vectors = np.append(vectors, vectors[-1:], axis=0)
            
            if randomize:
                self.interpolated_vectors,_, self.interpolated_vertices = self.randomize_positions_orientations(self.interpolated_vectors, tomorange)

            # print(sum(np.linalg.norm(vectors, axis=1)), self.length)
            # print (self.interpolated_vertices)

    def contour_nearest_distance(self, other_contour, interpolated = True, radius_offset = 0):
        if interpolated:
            vertices_self=self.interpolated_vertices
            vertices_other=other_contour.interpolated_vertices
            vectors_self = self.interpolated_vectors
            vectors_other=other_contour.interpolated_vectors
        else:
            vertices_self=self.original_vertices
            vertices_other=other_contour.original_vertices
            vectors_self = self.original_vectors
            vectors_other = other_contour.original_vectors
        if len(vertices_self) == 0 or len(vertices_other) == 0:
            return [10000, 10000, 10000], 10000, [30,60,90]
        dist_matrix = cdist(vertices_self, vertices_other)
        distance_vector=dist_matrix.min(
            axis = 1)-radius_offset
        neighbor_vectors = vectors_other[dist_matrix.argmin(axis=1)]
        # This is a huge performance bottleneck right here - need to vectorize!!!
        relative_orientation = [np.arccos(np.abs(np.dot(vectors_self[i], neighbor_vectors[i]))/(np.linalg.norm(vectors_self[i])*np.linalg.norm(neighbor_vectors[i])))*180/np.pi for i in range(len(vectors_self))]
        min_distance=min(distance_vector)
        return distance_vector, min_distance, relative_orientation
    
    def randomize_positions_orientations(self, vector, tomorange, rotrange=[-180, 180], tiltrange=[60,120]):
        """
        Return the same length vector, but with randomized reasonable orientation and fully randomized starting position. 
        Allows constrained rotations to keep general distribution similar.

        Used for generating random distributions with the same overall density. Does not account for excluded volume from organelles, which is standard in PySeg's similar routines.

        Randomizes rot and tilt - because of the way euler angles are selected, large tilt ranges may lead to unacceptably nonuniform sampling.

        Currently is only applied to interpolated vectors, as later interpolation is messed up by wrapping.
        """
        ## Calculate and correct current orientations
        tomorange = [i/10 for i in tomorange]
        initial_vector = vector[0]
        initial_theta = np.arctan2(initial_vector[1],initial_vector[0])*180/np.pi
        initial_phi = np.arccos(initial_vector[2]/np.linalg.norm(initial_vector))*180/np.pi
        correction_rotation = R.from_euler("ZYZ", [initial_theta, initial_phi,200], degrees=True)
        # print(correction_rotation.as_euler("ZYZ", degrees=True))
        correction_rotation = correction_rotation.inv()
        # print(correction_rotation.as_euler("ZYZ", degrees=True))
        # correction_rotation = R.align_vectors([0,0,1], list(initial_vector))
        corrected_vector = correction_rotation.apply(vector)
        # print(vector[0])       
        # print(corrected_vector[0])

        # Randomize angles and positions
        initial = np.random.rand(3)*tomorange # rand is from 0 to 1
        rot = np.random.rand()*(rotrange[1]-rotrange[0])+rotrange[0]
        tilt = np.random.rand()*(tiltrange[1]-tiltrange[0])+tiltrange[0]
        rotation = R.from_euler("ZY", [rot, tilt], degrees=True)
        rotated_vector = rotation.apply(corrected_vector,)
        rotated_positions = np.cumsum(rotated_vector, axis=0)+initial
        rotated_positions_wrapped = rotated_positions

        for i in range(3):
            rotated_positions_wrapped[:][i] = rotated_positions[:][i] % tomorange[i]
        return rotated_vector, rotated_positions, rotated_positions_wrapped


def load_from_modfile(filename):
    """Generate tomo object from IMOD .mod file"""
    import subprocess
    asciifile=subprocess.run(
        ["imodinfo", "-a", filename], check = True, capture_output = True).stdout.decode()
    model=None
    pixel_size=1
    model_range=None
    objects=None
    countours=None
    number_of_objects=None
    offsets=None
    angles=None
    lines=asciifile.split("\n")
    iterator=iter(lines)
    oldvertex=(0, 0, 0)
    for line in iterator:
        if line.startswith("#"):
            continue
        if line == '':
            continue
        if line.startswith("imod"):
            number_of_objects=int(line.split()[-1])
        if line.startswith("max"):
            model_range=(int(i) for i in line.split()[1:4])
        if line.startswith("offsets"):
            offsets=(float(i) for i in line.split()[1:4])
        if line.startswith("angles"):
            angles=(float(i) for i in line.split()[1:4])
        if line.startswith("pixsize"):
            pixel_size=float(line.split()[-1])
        if line.startswith("units"):
            # Assert(line.split()[-1]=="nm")
            model=TomoModel(filename, pixel_size, model_range, offsets)
        if line.startswith("object"):
            object_number, number_of_contours, number_of_meshes=(
                int(i) for i in line.split()[1:4])
            name=next(iterator).split()[-1]
            newobject=TomoObject(name)
            model.add_object(newobject)
            newline=next(iterator)
            while newline is not "":
                if newline.startswith("contour"):
                    contour_number, surface, n_vertices=[
                        int(i) for i in newline.split()[1:4]]
                    # print(object_number, contour_number, n_vertices)
                    vertices = []
                    for i in range(n_vertices):
                        newline = next(iterator)
                        vertex = tuple(
                            [float(i)*pixel_size for i in newline.split()])
                        if not oldvertex == vertex:
                            vertices.append(vertex)
                            oldvertex=vertex
                    newobject.add_contour(FilamentContour(
                        vertices, interpolate=True, interpolation_period_nm=INTERPOLATION_PERIOD))
                newline=next(iterator)

    return model






def load_amira_models(modelname, list_of_filenames, list_of_objectnames, pixel_size = 1, tomorange = (0, 0, 0), tomooffset = (0, 0, 0), randomize=False, interpolate=True, interpolation_period=INTERPOLATION_PERIOD):
    """Import a series of filament XML files from amira into a new tomomodel."""
    print(modelname)
    assert(len(list_of_filenames) == len(list_of_objectnames))
    model=TomoModel(modelname, pixel_size, tomorange, tomooffset)
    for index, name in enumerate(list_of_objectnames):
        newobject=load_xml_object(
            name, list_of_filenames[index], tomorange, tomooffset, randomize=randomize, interpolate=interpolate, interpolation_period=interpolation_period)
        model.add_object(newobject)
    return model


def load_xml_object(object_name, filename, tomorange, tomooffset, randomize=False, interpolate=True, interpolation_period=INTERPOLATION_PERIOD):
    xmlobject = ModelComponent(object_name)
    # print(filename)
    tree = ET.parse(filename)
    root = tree.getroot()

    points = root[2]
    segments = root[3]


    for segment in segments[0][1:]:
        # Most of this math is because amira uses an inverted view and does some weird offsetting stuff, so its to get the models in line with the real data.
        # print(segment[0][0].text)
        contour_list = []
        curved_length = float(segment[1][0].text)
        # print(curved_length)
        list_of_points = segment[-1][0].text.split(",")
        for point in list_of_points:
            point_id = int(point)
            point_xml = points[0][point_id+1]
            assert(int(point_xml[0][0].text) == point_id)
            # Divide by 10 to get to nm, not angstroms
            x = (float(point_xml[-3][0].text)-tomooffset[0])/10
            y = (tomorange[1] -
                 (float(point_xml[-2][0].text)-tomooffset[1]))/10
            z = (float(point_xml[-1][0].text)-tomooffset[2])/10
            data_tuple = (x, y, z)
            contour_list.append(data_tuple)
        xmlobject.add_contour(FilamentContour(
            contour_list, interpolate=interpolate, interpolation_period_nm=interpolation_period, randomize=randomize, tomorange=tomorange))
    print(len(segments[0][1:]))
    return xmlobject
