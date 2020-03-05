import numpy as np
from scipy.interpolate import splprep, splev
from matplotlib import pyplot as plt
from sys import argv
import glob


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
				tck,u = splprep([x,y,z], k=1)
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
		if len(vertices_self) == 0 or len(vertices_other) == 0:
			return [10000,10000,10000], 10000
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







	

model_folder = argv[1]
file_list = glob.glob("{}/*.mod".format(model_folder))
total_distances = []
for filename in file_list:
	prefix = filename.split("/")[-1].split(".")[0]
	print(prefix)
	model = load_from_modfile(filename)
	keys = model.get_objects().keys()
	if not "septin" in keys:
		print("Model {} does not have a septin object - skipping".format(filename))
		continue
	if len(model.get_object("septin").get_contours()) == 0:
		print("Model {} does not have any septin contours - skipping".format(filename))
		continue
	if not "microtubule" in keys:
		print("Model {} does not have a microtubule object - skipping".format(filename))
		continue
	if len(model.get_object("microtubule").get_contours()) == 0:
		print("Model {} does not have any microtubule contours - skipping".format(filename))
		continue
	total_distances_per_model = []

	for septin_count, contour in enumerate(model.get_object("septin").get_contours()):
		distances = [np.linalg.norm(np.subtract(contour.interpolated_vertices[i+1],contour.interpolated_vertices[i])) for i in range(20, len(contour.interpolated_vertices)-20)]
		# print(distances)
		# plt.title(septin_count+1)
		# plt.hist(distances, bins=30)
		# plt.show()
		min_distance = 10000
		nearest_tubule = None
		distance_vector = [10000,10000,10000]
		# distance_vector=[10000 for i in contour.interpolated_vertices]
		for tubule_position, tubule_contour in enumerate(model.get_object("microtubule").get_contours()):
			vector, distance = contour.contour_nearest_distance(tubule_contour)
			if distance < min_distance:
				min_distance = distance
				distance_vector = vector
				nearest_tubule = tubule_position
		if min_distance<100:
			total_distances.extend(distance_vector)
			total_distances_per_model.extend(distance_vector)
			print("Septin Number: ", septin_count+1)
			print("Nearest Tubule: ", nearest_tubule+1)
			print("Minimum distance: ", min_distance, "nm")
			fig, ax = plt.subplots()
			ax.plot(range(len(distance_vector)), distance_vector)
			ax.set_ylabel("Distance (nm)")
			ax.set_title("Distance between Septin {} and Tubule {}".format(septin_count+1, tubule_position+1))
			plt.tight_layout()
			fig.savefig("{}_{}.svg".format(prefix, septin_count+1))
			fig2,ax2=plt.subplots()
			ax2.set_title("Histogram of Distances between Septin {} and Tubule {}".format(septin_count+1, tubule_position+1))
			ax2.hist(distance_vector,bins=range(0, 100, 4))
			fig2.savefig("{}_{}_histogram.svg".format(prefix, septin_count+1))
			print(distance_vector)

	fig3,ax3 = plt.subplots()
	ax3.set_title("Histograms of septin-MT distances for all\n septins that come within 100nm of an MT")
	ax3.set_ylabel("Count of 1nm stretches of septin")
	ax3.set_xlabel("Septin-MT distance (nm)")
	ax3.hist(total_distances_per_model, bins=range(0, 100, 4))
	fig3.savefig("Total_histogram_{}.svg".format(prefix))

fig3,ax3 = plt.subplots()
ax3.set_title("Histograms of septin-MT distances for all\n septins that come within 100nm of an MT")
ax3.set_ylabel("Count of 1nm stretches of septin")
ax3.set_xlabel("Septin-MT distance (nm)")
ax3.hist(total_distances, bins=range(0, 100, 2))
fig3.savefig("Total_histogram.svg")
	# print(distance_vector)


