import copy
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

from os.path import isfile
from scipy.interpolate import splprep, splev
import xml.etree.ElementTree as ET

import model

my_cmap = copy.copy(matplotlib.cm.get_cmap('viridis')) # copy the default cmap
my_cmap.set_bad(my_cmap.colors[0])

FILTER_DISTANCE = 10000
FILTER_ORIENTATION = 100
FILTER = None # Set to a nm radius to filter on distances.
INTERPOLATE = True
INTERPOLATION_PERIOD = 4 #nm
# radius_MT = 11
# radius_septin = 5
RANDOMIZE = True

basename = "/Users/benjaminbarad/Dropbox (Scripps Research)/Grotjahn Lab/Data/OldData/FissionProject/CaltechDataset/"
# folders = ["May14_TS03/May14_TS3_xmlfiles/"]
# tomoranges = [(20774.395, 20081.914, 5409.9985)]


folders = ["Apr20_TS02/Apr20_TS02_xmlfiles/", "July29_TS05/July29_TS05_xmlfiles/", "July29_TS07/July29_TS07_xmlfiles/","July29_TS08/July29_TS08_xmlfiles/",
            "July29_TS09/July29_TS09_xmlfiles/","July29_TS11/June29_TS11_xmlfiles/", "July29_TS12/July29_TS12_xmlfiles/","July29_TS13/July29_TS13_xmlfiles/",
            "July29_TS14/July29_TS14_xmlfiles/","June11_TS01/June11_TS01_xmlfiles/","June11_TS04/June11_TS04_xmlfiles/","June11_TS08/June11_TS08_xmlfiles/",
            "June11_TS09/June11_TS09_xmlfiles/","May14_TS03/May14_TS3_xmlfiles/","Oct16_TS01/Oct16_TS01_xmlfiles/"]
tomoranges = [(20774.395, 20081.914, 8656.),(16588.8, 16035.84, 4838.4), (16588.8, 16035.84, 5184.),(16588.8, 16035.84, 5184.),
              (16588.8, 16035.84, 4320.),(16588.8, 16035.84, 4665.6),(16588.8, 16035.84, 4492.8),(16588.8, 16035.84, 4665.6),
              (16588.8, 16035.84, 4838.4),(20774.395, 20081.914, 7141.2),(20774.395, 20081.914, 6492.),(20774.395, 20081.914, 5410.),
              (20774.395, 20081.914, 6492.),(20774.395, 20081.914, 5410.),(16588.8, 16035.84, 4942.08)]
radii = {"septin":5, "actin": 3.5, "microtubule":12.5}
list_of_objectnames = ["septin", "actin", "microtubule"]
foldername = "/Users/benjaminbarad/Dropbox (Scripps Research)/Grotjahn Lab/Data/OldData/FissionProject/CaltechDataset/May14_TS03/May14_TS3_xmlfiles/"
# septin_filename = "TS3_4bin_SIRTl20_rotx2.CorrelationLines.am_septin.xml"
# mt_filename = "TS3_4bin_SIRTl20_rotx2.CorrelationLines_MTs.xml"
extent = (20774.395, 20081.914, 5409.9985)
offset = (0, 0, 0)
pixsizes = [21.64, 17.28, 17.28, 17.28, 17.28, 17.28, 17.28, 17.28, 17.28, 21.64, 21.64,21.64, 21.64, 21.64, 17.28]




def septin_mt_distances(amiramodel):
    near_mt_count_per_file = 0
    sum_distance_per_file = 0
    sum_distance_cooriented_per_file = 0
    sum_distance_nearby_per_file = 0
    sum_distance_cooriented_per_septin = 0
    prefix = foldername.split('/')[-2]
    print(prefix)
    keys = amiramodel.get_objects().keys()
    if not "septin" in keys:
        print("Model {} does not have a septin object - skipping".format(filename))
        return None
    if len(amiramodel.get_object("septin").get_contours()) == 0:
        print("Model {} does not have any septin contours - skipping".format(filename))
        return None
    septin_total_count_per_file = len(
        amiramodel.get_object("septin").get_contours())
    if not "microtubule" in keys:
        print("Model {} does not have a microtubule object - skipping".format(filename))
        return None
    if len(amiramodel.get_object("microtubule").get_contours()) == 0:
        print("Model {} does not have any microtubule contours - skipping".format(filename))
        return None
    microtubule_total_count_per_file = len(
        amiramodel.get_object("microtubule").get_contours())
    total_distances_per_model = []
    total_orientations_per_model = []
    for septin_count, contour in enumerate(amiramodel.get_object("septin").get_contours()):
        distances = np.linalg.norm(contour.interpolated_vectors, axis=1)
        sum_distance_per_septin = sum(distances)
        sum_distance_per_file += sum_distance_per_septin
        min_distance = 10000
        nearest_tubule = None
        orientation_vector = [30, 60, 90]
        distance_vector = [10000, 10000, 10000]
        # distance_vector=[10000 for i in contour.interpolated_vertices]
        for tubule_position, tubule_contour in enumerate(amiramodel.get_object("microtubule").get_contours()):
            # print(tubule_contour.interpolated_vertices)
            vector, distance, orientation = contour.contour_nearest_distance(
                tubule_contour, radius_offset=radius_MT+radius_septin)
            if distance < min_distance:
                min_distance = distance
                distance_vector = vector
                nearest_tubule = tubule_position
                orientation_vector = orientation
        if min_distance < FILTER_DISTANCE:
            near_mt_count_per_file += 1
            sum_distance_nearby_per_septin = sum(
                [j for i, j in enumerate(distances) if distance_vector[i] < FILTER_DISTANCE])
            sum_distance_cooriented_per_septin = sum(
                [j for i, j in enumerate(distances) if orientation_vector[i] < FILTER_ORIENTATION])
            sum_distance_nearby_per_file += sum_distance_per_septin
            sum_distance_cooriented_per_file += sum_distance_cooriented_per_septin
            # total_distances.extend(distance_vector)
            total_distances_per_model.extend(distance_vector)
            total_orientations_per_model.extend(orientation_vector)
            print("Septin Number: ", septin_count+1)
            print("Nearest Tubule: ", nearest_tubule+1)
            print("Minimum distance: ", min_distance, "nm")
            print("Minimum Relative angle: ", min(orientation_vector))
            print("Total Length of Septin: ", sum_distance_per_septin, "nm")
            print("Length of Septin within {}nm of MT {}: {}nm".format(
                FILTER_DISTANCE, nearest_tubule+1, sum_distance_nearby_per_septin))
            # fig, ax = plt.subplots()
            # ax.plot(distances_accrued, distance_vector)
            # ax.set_ylim(0,300)
            # ax.set_ylabel("Distance to microtubule (nm)")
            # ax.set_xlabel("Distance along septin filament (nm)")
            # ax.set_title("Distance between Septin {} and Tubule {}".format(septin_count+1, nearest_tubule+1))
            # plt.tight_layout()
            # fig.savefig("{}_{}.svg".format(prefix, septin_count+1))
            fig, ax = plt.subplots()
            ax.set_title("Histogram of Distances between Septin {} and Tubule {}".format(
                septin_count+1, nearest_tubule+1))
            ax.set_ylabel("Count of {}nm stretches of septin".format(
                INTERPOLATION_PERIOD))
            ax.set_xlabel("Septin-MT distance (nm)")
            print("{}nm out of {}nm total septin is within {}nm of microtubule {}".format(
                sum_distance_nearby_per_septin, sum_distance_per_septin, FILTER_DISTANCE, nearest_tubule+1))
            ax.hist(distance_vector, bins=range(0, 101, 4))
            ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            ax.set_xlabel("Distance (nm)")
            fig.savefig("{}_{}_histogram.svg".format(prefix, septin_count+1))

            fig2, ax2 = plt.subplots()
            ax2.set_title("Histogram of Relative Orientations between Septin {} and Tubule {}".format(
                septin_count+1, nearest_tubule+1))
            ax2.set_ylabel("Count of {}nm stretches of septin".format(
                INTERPOLATION_PERIOD))
            ax2.set_xlabel("Septin-MT Angle (º)")
            print("{}nm out of {}nm total septin is oriented less than {}º relative to microtubule {}".format(
                sum_distance_cooriented_per_septin, sum_distance_per_septin, FILTER_ORIENTATION, nearest_tubule+1))
            ax2.hist(orientation_vector, bins=range(0, 90, 5))
            ax2.set_xticks([0, 15,30,45,60,75,90])

            fig2.savefig("{}_{}_angle_histogram.svg".format(prefix, septin_count+1))

            # print(distance_vector)

        else:
            sum_distance_nearby_per_septin = 0
            print("Septin Number: ", septin_count+1)
            print("No MT within {}nm".format(FILTER_DISTANCE))

    fig3, ax3 = plt.subplots()
    ax3.set_title(
        "Histograms of septin-MT distances for all\n septins that come within {}nm of an MT".format(FILTER_DISTANCE))
    ax3.set_ylabel("Count of {}nm stretches of septin".format(
        INTERPOLATION_PERIOD))
    ax3.set_xlabel("Septin-MT distance (nm)")
    ax3.hist(total_distances_per_model, bins=range(0, 200, 4))
    # ax3.set_xticks([0, 20, 40, 0, 40, 50, 60, 70, 80, 90, 100])
    fig3.savefig("Total_histogram_{}.svg".format(prefix))
    plt.close('all')
    print("For Model {}, {} out of {} septins are within {}nm of one of the {} microtubules".format(
        prefix, near_mt_count_per_file, septin_total_count_per_file, FILTER_DISTANCE, microtubule_total_count_per_file))
    print("For Model {}, {}nm out of {}nm total septin are within {}nm of a microtubule".format(
        prefix, sum_distance_nearby_per_file, sum_distance_per_file, FILTER_DISTANCE))

    fig4, ax4 = plt.subplots()
    ax4.set_title("Histogram of Septin-MT relative angles for all\n septins that come within {}nm of an MT".format(FILTER_DISTANCE))
    ax4.set_ylabel("Count of {}nm stretches of septin".format(
        INTERPOLATION_PERIOD))
    ax4.set_xlabel("Septin-MT Angle (º)")
    print("For Model {}, {}nm out of {}nm total septin is oriented less than {}º relative to a microtubule".format(
        prefix, sum_distance_cooriented_per_file, sum_distance_per_file, FILTER_ORIENTATION))
    ax4.hist(total_orientations_per_model, bins=range(0, 90, 5))
    ax4.set_xticks([0, 15,30,45,60,75,90])

    fig4.savefig("{}_{}_angle_histogram.svg".format(prefix, septin_count+1))

def distances_orientations(amiramodel, key_1, key_2=None, randomize=False, radius_offset=0, filter=None, individual_plots=False):
    if not key_2:
        key_2 = key_1
        self_comparison = True
    else:
        self_comparison = False
    min_distance = 100000
    distances = []
    orientations = []
    for count, contour in enumerate(amiramodel.get_object(key_1).get_contours()):
        # distances = np.linalg.norm(contour.interpolated_vectors, axis=1)
        # sum_distance_per_septin = sum(distances)
        # sum_distance_per_file += sum_distance_per_septin
        min_distance = 100000
        nearest_contour = None
        orientation_vector = []
        distance_vector = []
        # distance_vector=[10000 for i in contour.interpolated_vertices]
        for count_2, contour_2 in enumerate(amiramodel.get_object(key_2).get_contours()):
            if self_comparison and count_2 == count:
                continue
            vector, distance, orientation = contour.contour_nearest_distance(
                contour_2, radius_offset=radius_offset)
            if distance < min_distance:
                min_distance = distance
                distance_vector = vector
                nearest_contour = count_2
                orientation_vector = orientation
        if filter and min_distance>filter:
            continue
        distances.append(distance_vector)
        orientations.append(orientation_vector)
        if individual_plots and nearest_contour:
            make_plots(amiramodel.name, distances, orientations, key_1+"_"+count, key_2+"_"+nearest_contour, randomize, filter=None, interpolate=INTERPOLATE, interpolation_period=INTERPOLATION_PERIOD)
    distances = np.concatenate(distances,axis=None)
    orientations = np.concatenate(orientations, axis=None)
    
    return distances, orientations

def make_plots(prefix, distances, orientations, key1, key2, randomize, filter, interpolate, interpolation_period, format=".svg"):
    # Distances histogram
    distances_filename = f"{prefix}_{key1}_{key2}_distances"
    distances_title = f"{prefix} {key1}-{key2} distance histogram"
    if filter:
        distances_filename+=(f"_{filter}nm")
        distances_title+=(f" for {key1} filaments within {filter}nm of any {key2}")
    if randomize:
        distances_filename+=("_randomized")
        distances_title+=(" (randomized positions and orientations)")
    distances_yaxis = f"Count of {interpolation_period}nm stretches of {key1}"
    distances_xaxis = f"{key1}-{key2} distance (nm)"  
    fig, ax = plt.subplots()
    ax.set_title(distances_title, wrap=True, pad=10)
    ax.set_ylabel(distances_yaxis)
    ax.set_xlabel(distances_xaxis)
    ax.hist(distances, bins=range(0, 201, 4))
    ax.set_xlim(0,200)
    ax.set_xticks(range(0,201,20))
    # ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # plt.tight_layout()
    fig.savefig(distances_filename+format)

    #Orientations
    orientations_filename = f"{prefix}_{key1}_{key2}_orientations"
    orientations_title = f"{prefix} {key1}-{key2} orientation histogram"
    if filter:
        orientations_filename+=f"_{filter}nm"
        orientations_title+=f" for {key1} filaments within {filter}nm of any {key2}"
    if randomize:
        orientations_filename+="_randomized"
        orientations_title+=" (randomized positions and orientations)"
    orientations_yaxis = f"Count of {interpolation_period}nm stretches of {key1}"
    orientations_xaxis = f"{key1}-{key2} orientation (º)"  
    fig2, ax2 = plt.subplots()
    ax2.set_title(orientations_title, wrap=True, pad=10)
    ax2.set_ylabel(orientations_yaxis)
    ax2.set_xlabel(orientations_xaxis)
    ax2.hist(distances, bins=range(0, 91, 5))
    ax2.set_xticks(range(0,91,15))
    ax2.set_xlim(0,90)
    # ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # plt.tight_layout()
    fig2.savefig(orientations_filename+format)

    # Distances vs Orientations
  

    twod_filename = f"{prefix}_{key1}_{key2}_2d_hist"
    twod_title = f"{prefix} {key1}-{key2} orientation vs distance heatmap"
    if filter:
        twod_filename += f"_{filter}nm"
        twod_title += f" for {key1} filaments within {filter}nm of any {key2}"
    if randomize:
        twod_filename+=("_randomized")
        twod_title+=(" (randomized positions and orientations)")
    axrange = [[0,200], [0,90]]
    fig3,ax3 = plt.subplots()
    # img = np.histogram2d(distances, orientations, bins=20, range=range)
    # im = ax3.imshow(img)
    _, _, _, im = ax3.hist2d(distances, orientations, bins=30, range=axrange, cmap=my_cmap)
    fig3.colorbar(im, ax=ax3, label=f"{interpolation_period}nm stretches of {key1} per bin")
    ax3.set_title(twod_title, wrap=True, pad=10)
    ax3.set_yticks(range(0,91,15))
    ax3.set_ylabel(f"{key1}-{key2} orientation (º)")
    ax3.set_xticks(range(0,201,20))
    ax3.set_xlabel(f"{key1}-{key2} distance (º)")
    # plt.tight_layout()
    fig3.savefig(twod_filename+format)

    fig4,ax4 = plt.subplots()
    # img = np.histogram2d(distances, orientations, bins=20, range=range)
    # im = ax3.imshow(img)
    _, _, _, im = ax4.hist2d(distances, orientations, bins=30, range=axrange, norm=matplotlib.colors.LogNorm(), cmap=my_cmap)
    fig4.colorbar(im, ax=ax4, label=f"{interpolation_period}nm stretches of {key1} per bin")
    ax4.set_title(twod_title, wrap=True, pad=10)
    ax4.set_yticks(range(0,91,15))
    ax4.set_ylabel(f"{key1}-{key2} orientation (º)")
    ax4.set_xticks(range(0,201,20))
    ax4.set_xlabel(f"{key1}-{key2} distance (º)")
    # plt.tight_layout()
    fig4.savefig(twod_filename+"_log"+format)





actin_actin_distances = []
actin_actin_orientations = []
septin_tubule_distances = []
septin_tubule_orientations = []
actin_tubule_distances = []
actin_tubule_orientations = []
septin_septin_distances = []
septin_septin_orientations = []
for index, folder in enumerate(folders):
    prefix = folder.split("/")[0]
    extent = tomoranges[index]
    offset = (0,0,0)
    filenames = []
    objectnames = []

    for index,object in enumerate(list_of_objectnames):
        filename = basename+folder+object+".xml"
        if isfile(filename):
            filenames.append(filename)
            objectnames.append(object)
    amiramodel = model.load_amira_models(
        prefix, filenames, objectnames, 1, extent, offset, interpolate = True, interpolation_period = INTERPOLATION_PERIOD, randomize=RANDOMIZE)

    if len(filenames) == 0:
        print("{} appears to have no associated files".format(prefix))
        continue
    if "septin" in objectnames:
        radius = radii["septin"]*2
        distance, orientation = distances_orientations(amiramodel, "septin", randomize=RANDOMIZE, radius_offset=radius, filter=FILTER)
        septin_septin_distances.append(distance)
        septin_septin_orientations.append(orientation)
        if len(distance)>0 and min(distance) < 200:
            make_plots(amiramodel.name, distance,orientation, "septin","septin",randomize=RANDOMIZE, filter=FILTER,interpolate=INTERPOLATE, interpolation_period=INTERPOLATION_PERIOD)
    if "actin" in objectnames and "microtubule" in objectnames:
        radius = radii["actin"]+radii["microtubule"]
        distance, orientation = distances_orientations(amiramodel, "actin", "microtubule", randomize=RANDOMIZE, radius_offset=radius, filter=FILTER)
        actin_tubule_distances.append(distance)
        actin_tubule_orientations.append(orientation)
        if len(distance)>0 and min(distance) < 200:
            make_plots(amiramodel.name, distance,orientation, "actin","microtubule",randomize=RANDOMIZE, filter=FILTER,interpolate=INTERPOLATE, interpolation_period=INTERPOLATION_PERIOD)
    if "septin" in objectnames and "microtubule" in objectnames:
        radius = radii["septin"]+radii["microtubule"]
        distance, orientation = distances_orientations(amiramodel, "septin", "microtubule", randomize=RANDOMIZE, radius_offset=radius, filter=FILTER)
        septin_tubule_distances.append(distance)
        septin_tubule_orientations.append(orientation)
        if len(distance)>0 and min(distance) < 200:
            make_plots(amiramodel.name, distance,orientation, "septin","microtubule",randomize=RANDOMIZE, filter=FILTER,interpolate=INTERPOLATE, interpolation_period=INTERPOLATION_PERIOD)
    if "actin" in objectnames:
        radius = radii["actin"]*2
        distance, orientation = distances_orientations(amiramodel, "actin", randomize=RANDOMIZE, radius_offset=radius, filter=FILTER)
        actin_actin_distances.append(distance)
        actin_actin_orientations.append(orientation)
        if len(distance)>0 and min(distance) < 200:
            make_plots(amiramodel.name, distance,orientation, "actin","actin",randomize=RANDOMIZE, filter=FILTER,interpolate=INTERPOLATE, interpolation_period=INTERPOLATION_PERIOD)

septin_septin_distances = np.concatenate(septin_septin_distances, axis=None)
septin_septin_orientations = np.concatenate(septin_septin_orientations, axis=None)
make_plots("Total", septin_septin_distances,septin_septin_orientations, "septin","septin",randomize=RANDOMIZE, filter=FILTER,interpolate=INTERPOLATE, interpolation_period=INTERPOLATION_PERIOD)

septin_tubule_distances = np.concatenate(septin_tubule_distances, axis=None)
septin_tubule_orientations = np.concatenate(septin_tubule_orientations, axis=None)
make_plots("Total", septin_tubule_distances,septin_tubule_orientations, "septin","microtubule",randomize=RANDOMIZE, filter=FILTER,interpolate=INTERPOLATE, interpolation_period=INTERPOLATION_PERIOD)

actin_tubule_distances = np.concatenate(actin_tubule_distances, axis=None)
actin_tubule_orientations = np.concatenate(actin_tubule_orientations, axis=None)
make_plots("Total", actin_tubule_distances,actin_tubule_orientations, "actin","microtubule",randomize=RANDOMIZE, filter=FILTER,interpolate=INTERPOLATE, interpolation_period=INTERPOLATION_PERIOD)

actin_actin_distances = np.concatenate(actin_actin_distances, axis=None)
actin_actin_orientations = np.concatenate(actin_actin_orientations, axis=None)
make_plots("Total", actin_actin_distances,actin_actin_orientations, "actin","actin",randomize=RANDOMIZE, filter=FILTER,interpolate=INTERPOLATE, interpolation_period=INTERPOLATION_PERIOD)



# amiramodel = model.load_amira_models(
#     "May14_TS03", list_of_filenames, list_of_objectnames, pixsize, extent, offset, randomize=RANDOMIZE)
# septin_mt_distances(amiramodel)
