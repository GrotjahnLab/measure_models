import numpy as np
from scipy.interpolate import splprep, splev
from matplotlib import pyplot as plt


class IMODModel():
	"""Top level model class for data from IMOD"""
	def __init__(self, name, pixel_size, model_range_tuple, offset_tuple):
		self.name = name
		self.number_of_objects=0
		self.pixel_size = pixel_size
		self.model_range = model_range_tuple
		self.offset = offset_tuple
		self.objects = {}

	def add_object(self, imodobject):
		self.number_of_objects +=1
		self.objects[imodobject.name] = imodobject

	def get_objects(self):
		return self.objects

	def get_object(self, name):
		return self.objects[name]


class IMODObject():
	"""Object class for data from IMOD"""
	def __init__(self, name):
		self.name = name
		self.number_of_contours=0
		self.contours = []

	def add_contour(self, imodcontour):
		self.number_of_contours += 1
		self.contours.append(imodcontour)

	def get_contours(self):
		return self.contours


class IMODContour():
	"""Contour class for data from IMOD"""
	def __init__(self, vertices, interpolate=True, interpolation_period_nm=1):
		self.original_vertices=vertices
		self.calculate_length_nm()
		if interpolate:
			self.interpolate_vertices(interpolation_period_nm)

	def calculate_length_nm(self):
		if len(self.original_vertices)<2:
			self.length=0
		else:
			length = 0
			for i in range(len(self.original_vertices)-1):
				length += np.linalg.norm(np.subtract(self.original_vertices[i+1],self.original_vertices[i]))
			self.length = length
		return self.length

	def interpolate_vertices(self, interpolation_period_nm):
		self.interpolation_period_nm = interpolation_period_nm
		if len(self.original_vertices) < 2:
			self.interpolated_vertices = self.original_vertices
		else:
			# TODO - FIGURE OUT HOW TO MAKE THIS NOT HAVE ROUNDING ERRORS
			x,y,z = zip(*self.original_vertices)
			if len(x) > 3:
				tck,u = splprep([x,y,z], k=3)
			else:
				tck,u = splprep([x,y,z], k=1)
			
			u_fine = np.linspace(u[0],u[-1],int(self.length/interpolation_period_nm))
			x_fine,y_fine,z_fine=splev(u_fine,tck)
			self.interpolated_vertices = list(zip(x_fine,y_fine,z_fine))
			# print (self.interpolated_vertices)

	def contour_nearest_distance(self, other_contour, interpolated=True):
		if interpolated:
			vertices_self = self.interpolated_vertices
			vertices_other = other_contour.interpolated_vertices
		else:
			vertices_self = self.original_vertices
			vertices_other = other_contour.original_vertices
		distance_vector = [min([np.linalg.norm(np.subtract(i,j)) for j in vertices_other]) for i in vertices_self]
		min_distance = min(distance_vector)
		return distance_vector, min_distance

def load_from_modfile(filename):
	import subprocess
	asciifile = subprocess.run(["imodinfo", "-a", filename], check=True, capture_output=True).stdout.decode()
	model = None
	pixel_size = 1
	model_range = None
	objects = None
	countours = None
	number_of_objects = None
	offsets=None
	angles = None
	lines = asciifile.split("\n")
	iterator = iter(lines)
	oldvertex=(0,0,0)
	for line in iterator:
		if line.startswith("#"):
			continue
		if line == '':
			continue
		if line.startswith("imod"):
			number_of_objects = int(line.split()[-1])
		if line.startswith("max"):
			model_range = (int(i) for i in line.split()[1:4])
		if line.startswith("offsets"):
			offsets = (float(i) for i in line.split()[1:4])
		if line.startswith("angles"):
			angles = (float(i) for i in line.split()[1:4])
		if line.startswith("pixsize"):
			pixel_size = float(line.split()[-1])
		if line.startswith("units"):
			# Assert(line.split()[-1]=="nm")
			model = IMODModel(filename, pixel_size, model_range, offsets)
		if line.startswith("object"):
			object_number, number_of_contours, number_of_meshes = (int(i) for i in line.split()[1:4])
			name = next(iterator).split()[-1]
			newobject = IMODObject(name)
			model.add_object(newobject)
			newline = next(iterator)
			while newline is not "":
				if newline.startswith("contour"):
					contour_number, surface, n_vertices = [int(i) for i in newline.split()[1:4]]
					# print(object_number, contour_number, n_vertices)
					vertices = []
					for i in range(n_vertices):
						newline = next(iterator)
						vertex = tuple([float(i)*pixel_size for i in newline.split()])
						if not oldvertex == vertex:
							vertices.append(vertex)
							oldvertex = vertex
					newobject.add_contour(IMODContour(vertices, interpolate=True, interpolation_period_nm=1))
				newline = next(iterator)

	return model







	

model = load_from_modfile("_data/July29_TS7_SeptinMT.mod")
for septin_count, contour in enumerate(model.get_object("septin").get_contours()):
	min_distance = 10000
	nearest_tubule = None
	# distance_vector=[10000 for i in contour.interpolated_vertices]
	for tubule_position, tubule_contour in enumerate(model.get_object("microtubule").get_contours()):
		vector, distance = contour.contour_nearest_distance(tubule_contour)
		if distance < min_distance:
			min_distance = distance
			distance_vector = vector
			nearest_tubule = tubule_position
	if min_distance<100:
		print("Septin Number: ", septin_count)
		print("Nearest Tubule: ", nearest_tubule)
		print("Minimum distance: ", min_distance, "nm")
		fig, ax = plt.subplots()
		ax.plot(range(len(distance_vector)), distance_vector)
		ax.set_ylabel("Distance (nm)")
		ax.set_title("Distance between Septin {} and Tubule {}".format(septin_count, tubule_position))
		plt.tight_layout()
		fig.savefig("{}.png".format(septin_count))
		print(distance_vector)

	# print(distance_vector)


