import sys
import numpy as np
from scipy.spatial import Delaunay, distance_matrix
import matplotlib.pyplot as plt
import random

import read_dump as rd
import math_lib as ml
import dim_reduc as dm

def tesselate(frames, dims, xcol=3, ycol=4, skin=0.5):
    """
    input is a 2D (i.e. many xyz coordinates) or 3D (i.e. multiple frames of many xyz coordinates) np.array
    tessalates system in x and y, then returns points within skin*box_length of box dimensions
    """
    if len(frames.shape) == 2:
        n_atoms, n_fields = frames.shape
        out = np.zeros((n_atoms*9, n_fields))
        counter = 0
        for ii in range(-1,2):
            for jj in range(-1,2):
                    shift_frames = np.copy(frames)
                    shift_frames[:,xcol] += ii*(dims[0,1] - dims[0,0])
                    shift_frames[:,ycol] += jj*(dims[1,1] - dims[1,0])
                    out[n_atoms*counter:n_atoms*(counter+1), :] = shift_frames

                    counter += 1
        out = np.array([p for p in out if p[0] > box_dims[0,0] - (box_dims[0,1] - box_dims[0,0])*skin
                                                   and p[0] < box_dims[0,1] + (box_dims[0,1] - box_dims[0,0])*skin
                                                   and p[1] > box_dims[1,0] - (box_dims[1,1] - box_dims[1,0])*skin
                                                   and p[1] < box_dims[1,1] + (box_dims[1,1] - box_dims[1,0])*skin])

        return out
    elif len(frames.shape) == 3:
        n_frames, n_atoms, n_fields = frames.shape
        out = np.zeros((n_frames, n_atoms*9, n_fields))
        counter = 0
        for ii in range(-1,2):
            for jj in range(-1,2):
                    for kk in range(n_frames):

                        shift_frames = np.copy(frames[kk])
                        shift_frames[:, xcol] += ii*(dims[kk,0,1] - dims[kk,0,0])
                        shift_frames[:, ycol] += jj*(dims[kk,1,1] - dims[kk,1,0])
                        out[kk, n_atoms*counter:n_atoms*(counter+1), :] = shift_frames

                    counter += 1

        points = np.array([p for p in tesse_points if p[0] > box_dims[0,0,0] - (box_dims[0,0,1] - box_dims[0,0,0])*skin
                                                   and p[0] < box_dims[0,0,1] + (box_dims[0,0,1] - box_dims[0,0,0])*skin
                                                   and p[1] > box_dims[0,1,0] - (box_dims[0,1,1] - box_dims[0,1,0])*skin
                                                   and p[1] < box_dims[0,1,1] + (box_dims[0,1,1] - box_dims[0,1,0])*skin])
        return out

def distance_filter_multi(frames, cutoff=100, nearest_neighbor=1):
    """
    distance filter for multiple frames
    checks at the end if the mask for each frame is the same and throws error if not
    """
    n_frames, n_atoms, n_fields = frames.shape
    masks = []
    for kk in range(n_frames):
        dist_matrix = distance_matrix(frames[kk], frames[kk])
        mask = np.ones(n_atoms, dtype=bool)
        for ii, distances in enumerate(dist_matrix):
            #sorts for nearest neighbor + itself
            idx = np.argpartition(distances, nearest_neighbor+1)
            if (distances[idx[:nearest_neighbor+1]] > cutoff).any():
                mask[ii] = False
        masks.append(mask)
    for kk in range(1, n_frames):
        if (-1 or 1) in (masks[kk].astype(int) - mask[0].astype(int)):
            print(kk)
            print([(ii,a) for ii, a in enumerate(masks[kk].astype(int) - mask[0].astype(int)) if a != 0])

    return frames[:, mask[0]]

def dist_filter(frame, cutoff=100, nearest_neighbor=1):
    """
    distance filter for a single frame
    """
    n_atoms, n_fields = frame.shape
    dist_matrix = distance_matrix(frame, frame)
    mask = np.ones(n_atoms, dtype=bool)
    for ii, distances in enumerate(dist_matrix):
        #sorts for nearest neighbor + itself
        idx = np.argpartition(distances, nearest_neighbor+1)
        if (distances[idx[:nearest_neighbor+1]] > cutoff).any():
            mask[ii] = False

    return mask, frame[mask]

def gen_random_points(n_points=5000):
    """
    generate test points with a simple functional form
    """
    temp_xyz = []
    dims = np.array([[0, 2*np.pi], [0, 2*np.pi], [-1,1]])
    for counter in range(n_points):
        x = random.uniform(dims[0,0], dims[0,1])
        y = random.uniform(dims[0,0], dims[0,1])
        temp_xyz.append([x,y,(np.cos(x)*np.sin(y))])
    points = np.array(temp_xyz)
    return points, dims

def gen_grid_points(dim=np.pi/20):
    """
    generate test points with simple function form on xy grid
    """
    xyz = []
    dims = np.array([[0, 2*np.pi], [0, 2*np.pi], [-15,15]])
    xy = np.mgrid[dims[0,0]:dims[0,1]:dim, dims[1,0]:dims[1,1]:dim].reshape(2,-1).T
    for x, y in xy:
        xyz.append([x,y,(np.cos(x)*np.sin(y))])
    xyz = np.array([xyz]).astype(float)
    points = np.array(xyz)
    out_points = np.array([ [1, a[0], a[1], a[2]] for a in xyz[0]])
    np.savetxt('grid_points.xyz', out_points, fmt='%i %f %f %f', header='%i\nCOMMMENT' % len(xyz[0]), comments='')
    return points, [dims]

def scale_points(points, dims):
    """
    scale coordinates to 0 to 1 based on some box dimensions
    return scaled points and scaled boxdimensions
    """
    scaled_points = np.copy(points)
    for ii in range(len(scaled_points)):
        for jj in range(len(scaled_points[ii])):
            scaled_points[ii, jj] -= dims[jj,0]
            scaled_points[ii, jj] /= (dims[jj,1] - dims[jj,0])
            #check that point is wrapped correctly as well
            if scaled_points[ii, jj] > 1:
                scaled_points[ii, jj] -= 1
                print("point %i isn't wrapped" % ii)
            elif scaled_points[ii, jj] < 0:
                scaled_points[ii, jj] += 1
                print("point %i isn't wrapped" % ii)
    scaled_dims = np.array([[0,1]]*len(scaled_points[0]))
    return scaled_points, scaled_dims

def unscale_points(coords, dims):
    """
    unscale coordinates from 0 to 1 to new box dimension
    return unscaled coordinates
    """
    if type(coords) == np.ndarray:
        unscaled_points = np.copy(coords)
        for ii in range(len(unscaled_points)):
            for jj in range(3):
                unscaled_points[ii, jj] *= (dims[jj,1] - dims[jj,0])
                unscaled_points[ii,jj] += dims[jj,0]
        return unscaled_points
    else:
        unscaled_points = {}
        for key, scaled_point in coords.items():
            if scaled_point.size != 0:
                unscaled_point = np.zeros(3)
                for jj in range(3):
                    unscaled_point[jj] = np.array(scaled_point[jj]) * (dims[jj,1] - dims[jj,0])
                    unscaled_point[jj] += dims[jj,0]
                unscaled_points[key] = unscaled_point
        return unscaled_points

class Voxel:
    """
    Voxel class, mirroring kmeans class
    average positions, labels, neighbors
    """

    ## occupied voxels
    centers = {}
    neighbors = {}
    labels = []

    shape = [0, 0, 0]
    ## initial voxels
    grid = []
    grid_neighbors = []

    def __init__(self, points, dims, voxel_dim=(1./50.), scaled=True, nofilter=True):
        """
        Initialize by scaling input points, voxel filtering, then unscaling
        """
        scaled_points, scaled_dims = scale_points(points, dims)
        print("voxel dim %f" % voxel_dim)
        self.voxel_filter(scaled_points, scaled_dims, voxel_dim=voxel_dim)
        self.centers = unscale_points(self.centers, dims)

    def voxel_filter(self, points, dims, voxel_dim=(1./50.)):
        """
        assign 3D grid, determine which point belong in each grid, then average their coordinates
        return average coordinates
        """
        self.grid =  np.mgrid[dims[0,0]:dims[0,1]:voxel_dim, dims[1,0]:dims[1,1]:voxel_dim, dims[2,0]:dims[2,1]:voxel_dim].reshape(3,-1).T
        print(self.grid)
        n_grid_points = len(self.grid)
        self.labels = np.zeros(len(points))
        self.occupant_positions = [ [] for ii in range(n_grid_points)]
        self.occupants = { ii:[] for ii in range(n_grid_points)}
        self.centers = { ii:[] for ii in range(n_grid_points)}
        self.shape = [int((dims[0,1]-dims[0,0])/voxel_dim), int((dims[1,1]-dims[1,0])/voxel_dim), int((dims[2,1]-dims[2,0])/voxel_dim)]
        print("occupants", len(self.occupants))
        for ii, point in enumerate(points):
            #print(point)
            #print(point[0]/voxel_dim, point[1]/voxel_dim, point[2]/voxel_dim)
            index0 = int(point[0]/voxel_dim) % self.shape[0]
            index1 = int(point[1]/voxel_dim) % self.shape[1]
            index2 = int(point[2]/voxel_dim) % self.shape[2]
            #print(index0, index1, index2)
            index = index0*self.shape[1]*self.shape[2]+index1*self.shape[2]+index2
            #print(index, index % (self.shape[0] * self.shape[1] * self.shape[2]))
            self.labels[ii] = index
            self.occupants[index].append(ii)
            self.occupant_positions[index].append(point)

        for ii, occu in enumerate(self.occupant_positions):
            #check if list is empty
            if occu:
                temp_voxel = np.array(occu)
                self.centers[ii] = np.average(temp_voxel, axis=0)
            else:
                self.centers[ii] = np.array([])
        self.centers[ii] = np.array(self.centers[ii])

        self.grid_neighbors = []
        for num in range(n_grid_points):
            x = int(num/(self.shape[1]*self.shape[2]))
            y = int(num%(self.shape[1]*self.shape[2]) / self.shape[1])
            z = int(num%(self.shape[1]*self.shape[2]) % self.shape[2])
            neighbor = []
            for ii in range(-1,2):
                for jj in range(-1,2):
                    for kk in range(-1,2):
                        index = ((x+ii)%self.shape[0])*self.shape[1]*self.shape[2]+((y+jj)%self.shape[2])*self.shape[1]+(z+kk)%self.shape[2]
                        if index > n_grid_points:
                            index = n_grid_points-index
                        neighbor.append(index)
            self.grid_neighbors.append(neighbor)

        self.neighbors = {}
        for num, neighs in enumerate(self.grid_neighbors):
            if self.occupants[num]:
                self.neighbors[num] = []
                for neigh in neighs:
                    if self.occupants[neigh]:
                        self.neighbors[num].append(neigh)
def get_one_area(a, b, c):
    """
    calculate the area of a single triangle
    """

    ab = b - a
    ac = c - a
    area = 0.5 * np.linalg.norm(np.cross(ab,ac))

    return area

def get_tri_area(points, tri):
    """
    calculate the area of all triangles in the triangulation
    """
    n_points = len(points)
    tri_areas = np.zeros(tri.simplices.shape[0])
    for ii, simplex in enumerate(tri.simplices):
    #    #shoelace formula
    #    tri_areas[ii] = 0.5 * np.abs(np.linalg.det(np.array([points[simplex[0]],
    #                                            points[simplex[1]],
    #                                            points[simplex[2]]])))
        #different formula
        ab = points[simplex[1]] - points[simplex[0]]
        ac = points[simplex[2]] - points[simplex[0]]
        tri_areas[ii] = 0.5 * np.linalg.norm(np.cross(ab,ac))
    if any(tri_areas) <= 0:
        print(tri_areas)
    
    return tri_areas

def get_cot_angle(x, y, z, verbose=False):
    """
    calculate the angle made by yx and zx
    """
    a = y - x
    a /= np.linalg.norm(a)

    b = z - x
    b /= np.linalg.norm(b)

    cot_angle = np.dot(a, b) / np.linalg.norm(np.cross(a, b))

    if verbose:
        print( "a", a)
        print( "b", b)
        print( "cot_angle", cot_angle)
    return cot_angle

def get_curvature(points, tri, subset=None):
    """
    Calculate the mean and gaussian curvature of a set of points
    Use the triangulation to determine the neighbors of each point
    """
    n_points = len(points)
    areas = np.zeros(n_points)
    areas_counter = np.zeros(n_points)
    c_angles = np.zeros((n_points, n_points))
    distances = np.zeros((n_points, n_points))

#    #calculate area of each triangle in the triangulation
#    tri_areas = get_tri_area(points, tri)
    for ii, simplex in enumerate(tri.simplices):
        in_subset = [index for index in simplex if index in subset]
        if not in_subset:
            continue
        tri_coords = np.copy(points[simplex])
        center = np.average(tri_coords, axis=0)
        #one vertex
        AB = (tri_coords[1] - tri_coords[0])/2 + tri_coords[0]
        AC = (tri_coords[2] - tri_coords[0])/2 + tri_coords[0]
        areas[simplex[0]] += get_one_area(tri_coords[0], AB, center)
        areas[simplex[0]] += get_one_area(tri_coords[0], AC, center)
        #second vertex
        BA = (tri_coords[0] - tri_coords[1])/2 + tri_coords[1]
        BC = (tri_coords[2] - tri_coords[1])/2 + tri_coords[1]
        areas[simplex[1]] += get_one_area(tri_coords[1], BA, center)
        areas[simplex[1]] += get_one_area(tri_coords[1], BC, center)
        #third
        CA = (tri_coords[0] - tri_coords[2])/2 + tri_coords[2]
        CB = (tri_coords[1] - tri_coords[2])/2 + tri_coords[2]
        areas[simplex[2]] += get_one_area(tri_coords[2], CA, center)
        areas[simplex[2]] += get_one_area(tri_coords[2], CB, center)

    #calculate mean curvature
    #1/(2*area) * sum( (cot(alpha) + cot(beta))* (x_j -x_i))
    #area is the area around the point of interest or 1/3 the area of all triangles that the point of interest is a vertex
    #alpha and beta are the angle opposite the line between point of interest and one of its nearest neighbors
    for ii, simplex in enumerate(tri.simplices):

        #check that one of the indices is in the subset of points to calculate curvature of
        in_subset = [index for index in simplex if index in subset]
        if not in_subset:
            continue

        for jj, index1 in enumerate(simplex[:-1]):
            for kk, index2 in enumerate(simplex[jj+1:]):

                vert1 = np.copy(points[index1])
                vert2 = np.copy(points[index2])
                distance = np.linalg.norm(vert2 - vert1)
                distances[index1, index2] += distance/2
                distances[index2, index1] += distance/2

        #redo for other permutations
        #calcualing the cot angle for each angle in the triangle
        obtuse = np.zeros(3).astype(bool)

        for jj in range(3):
            id1 = jj
            id2 = (jj+1) % 3
            id3 = (jj+2) % 3
            cot_angle = get_cot_angle(points[simplex[id1]], points[simplex[id2]], points[simplex[id3]])

            if cot_angle <= 0:
                obtuse[id1] = True

            c_angles[simplex[id2], simplex[id3]] += cot_angle
            c_angles[simplex[id3], simplex[id2]] += cot_angle
            
            #print simplex[id2], simplex[id3]
            #if (simplex[id2] == 18 and simplex[id3] == 21) or (simplex[id3] == 18 and simplex[id2] == 21):
            #    counter+=1
            #    print counter
            #    cot_angle = get_cot_angle(points[simplex[id1]], points[simplex[id2]], points[simplex[id3]], verbose=True)
            #    print(distances[18])
            #    print(c_angles[18])

            ###this is an bad fix to an error
            ##if c_angles[simplex[id2], simplex[id3]] != 0:
            ##    c_angles[simplex[id2], simplex[id3]] -= cot_angle
            ##else:
            ##    c_angles[simplex[id2], simplex[id3]] += cot_angle

            ##if c_angles[simplex[id3], simplex[id2]] != 0:
            ##    c_angles[simplex[id3], simplex[id2]] -= cot_angle
            ##else:
            ##    c_angles[simplex[id3], simplex[id2]] += cot_angle

#        #attribute area of each triangle to the vertex
#        #accounting for obtuse angles
#        areas_counter[simplex] += 1
#        if True in obtuse:
#            if np.sum(obtuse.astype(int)) > 1:
#                print(ii, simplex)
#                sys.exit("multiple obtuse angles in a single triangle")
#            if obtuse[0]:
#                areas[simplex[0]] += tri_areas[ii]/2
#                areas[simplex[1]] += tri_areas[ii]/4
#                areas[simplex[2]] += tri_areas[ii]/4
#            if obtuse[1]:
#                areas[simplex[0]] += tri_areas[ii]/4
#                areas[simplex[1]] += tri_areas[ii]/2
#                areas[simplex[2]] += tri_areas[ii]/4
#            if obtuse[2]:
#                areas[simplex[0]] += tri_areas[ii]/4
#                areas[simplex[1]] += tri_areas[ii]/4
#                areas[simplex[2]] += tri_areas[ii]/2
#        else:
#            areas[simplex] += tri_areas[ii]/3

    ##calculate total angle around each point
    ang_total = np.zeros(n_points)
    for ii, simplex in enumerate(tri.simplices):
        #check that one of the indices is in the subset of points to calculate curvature of
        in_subset = [index for index in simplex if index in subset]
        if not in_subset:
            continue

        xx = points[simplex[0]]
        yy = points[simplex[1]]
        zz = points[simplex[2]]

        ang_total[simplex[0]] += ml.Angle_3D(zz, xx, yy)
        ang_total[simplex[1]] += ml.Angle_3D(xx, yy, zz)
        ang_total[simplex[2]] += ml.Angle_3D(yy, zz, xx)

    #calculate mean curvature
    #calculate gaussian curvature from total angle and area
    m_curv = np.zeros(n_points)
    g_curv = np.zeros(n_points)
    if subset:
        for ii in subset:
            m_curv[ii] = np.dot(c_angles[ii], distances[ii]) / (2 * areas[ii])
            g_curv[ii] = (2*np.pi - ang_total[ii])/(areas[ii])
    else:
        for ii in range(n_points):
            m_curv[ii] = np.dot(c_angles[ii], distances[ii]) / (2 * areas[ii])
            g_curv[ii] = (2*np.pi - ang_total[ii])/(areas[ii])

    return m_curv, g_curv, c_angles, distances, ang_total, areas

#each case should define frames [n_frames, n_atoms, n_dimension] and box_dims [n_frames, [[hi, lo] * n_dimension]
#                        optionally define suffix
run_cases = ['random_coords', 'grid_coords', 'protrusion', 'test_protrusion', 'test_alex', 'test_MDA_leaflet']
run_case = sys.argv[1]

#optional
suffix = ''
dist_filter_cutoff = 50
dist_filtering = True
clustering = False
voxel_size=(1./25.)

if run_case == 'random_coords':
    ##test coordinates
    frames, box_dims = gen_random_points()

elif run_case == 'grid_coords':
    suffix = '_grid'
    frames, box_dims = gen_grid_points()

elif run_case == 'test_protrusion':
    ##sliced protrusion
    suffix = sys.argv[2]
    ## read all dumps
    timesteps, box_dims, frames_raw = rd.read_dump('../protrusion'+suffix+'.lammpstrj') #test.lammpstrj') #protrusion.lammpstrj')

    #get type 2
    frames = []
    frames_types = []
    for frame_raw in frames_raw:
        frame = []
        for atom in frame_raw:
            if atom[1] == 2:
                frame.append(np.array(atom))
        frame = np.array(frame)
        frames.append(frame)
    frames = np.array(frames)

    frames_types = frames[:, :, 1]
    frames = frames[:, :, 3:6]
    
    dist_filter_cutoff = 50
    dist_filtering = True
    clustering = True

elif run_case == 'protrusion':
    ##sliced protrusion
    suffix = sys.argv[2]
    ## read all dumps
    timesteps, box_dims, frames_raw = rd.read_dump('protrusion'+suffix+'.lammpstrj') #test.lammpstrj') #protrusion.lammpstrj')

    #get type 2
    frames = []
    frames_types = []
    for frame_raw in frames_raw:
        frame = []
        for atom in frame_raw:
            if atom[1] == 2:
                frame.append(np.array(atom))
        frame = np.array(frame)
        frames.append(frame)
    frames = np.array(frames)

    frames_types = frames[:, :, 1]
    frames = frames[:, :, 3:6]
    
    dist_filter_cutoff = 50
    dist_filtering = True
    clustering = True

elif run_case == 'protrusionTwoComp':
    ##sliced protrusion
    suffix = sys.argv[2]
    ## read all dumps
    timesteps, box_dims, frames_raw = rd.read_dump('protrusion'+suffix+'.lammpstrj') #test.lammpstrj') #protrusion.lammpstrj')

    #get type 2
    frames = []
    frames_types = []
    for frame_raw in frames_raw:
        frame = []
        for atom in frame_raw:
            if atom[1] == 2 or atom[1] == 3:
                frame.append(np.array(atom))
        frame = np.array(frame)
        frames.append(frame)
    frames = np.array(frames)

    frames_types = frames[:, :, 1]
    frames = frames[:, :, 3:6]
    
    dist_filter_cutoff = 50
    dist_filtering = True
    clustering = True

else:
    sys.exit('run_case not defined')

if dist_filtering:
    """
    loop frames, loop voxel, distance filter
    output new coords in list of np.array
    """
    new_frames = []
    for frame_num, frame in enumerate(frames):
        dist_filter_frame = []
        box_dim = np.copy(box_dims[frame_num])

        voxel = Voxel(frame, box_dim, voxel_dim=voxel_size)

        for index, occup in voxel.occupants.items():
            if index % 500 == 0:
                print("filtering %i of %i" % (index, len(voxel.occupants)))

            if not occup:
                continue

            group_ids = []
            group_xyz = np.empty((0,3))
            for neigh in voxel.neighbors[index]:
                group_ids += voxel.occupants[neigh]
            
                temp_xyz = np.copy(frame[voxel.occupants[neigh],:])

                if (voxel.centers[index][0] - voxel.centers[neigh][0]) > (box_dim[0,1] - box_dim[0,0])/2:
                    temp_xyz[:,0] += (box_dim[0,1] - box_dim[0,0])
                elif (voxel.centers[index][0] - voxel.centers[neigh][0]) < -(box_dim[0,1] - box_dim[0,0])/2:
                    temp_xyz[:,0] -= (box_dim[0,1] - box_dim[0,0])
                if (voxel.centers[index][1] - voxel.centers[neigh][1]) > (box_dim[1,1] - box_dim[1,0])/2:
                    temp_xyz[:,1] += (box_dim[1,1] - box_dim[1,0])
                elif (voxel.centers[index][1] - voxel.centers[neigh][1]) < -(box_dim[1,1] - box_dim[1,0])/2:
                    temp_xyz[:,1] -= (box_dim[1,1] - box_dim[1,0])
                group_xyz = np.append(group_xyz, temp_xyz,  axis=0)

            #skip voxels with less than 5 particles
            if len(group_xyz) < 5:
                continue

            #now implement distance filter, only on the local voxels
            group_mask, group_xyz_dist = dist_filter(group_xyz)

            #new indices of atoms in the voxel of interest
            center_group_ids = []
            for ii, num in enumerate(group_ids):
                if (num in voxel.occupants[index]) and (group_xyz[ii] in group_xyz_dist):
                    center_group_ids.append(ii)
            if len(center_group_ids) < 5:
                #print(index, len(center_group_ids), len(occup))
                continue
            if len(dist_filter_frame) == 0:
                dist_filter_frame = np.copy(group_xyz_dist[center_group_ids])
            else:
                dist_filter_frame = np.vstack((dist_filter_frame, group_xyz[center_group_ids])) 

        new_frames.append(dist_filter_frame)

    frames = new_frames

if clustering:
    """
    loop frames, loop voxel, cluster in each voxel
    output new coords in list of np.array
    """
    from sklearn.cluster import KMeans
    new_frames = []
    cluster_statistics = []
    for frame_num, frame in enumerate(frames):
        cluster_frame = []
        box_dim = np.copy(box_dims[frame_num])

        voxel = Voxel(frame, box_dim, voxel_dim=voxel_size)

        for index, occup in voxel.occupants.items():
            if index % 500 == 0:
                print("clustering %i of %i" % (index, len(voxel.occupants)))
            if not occup:
                continue
            n_clusters = int(len(occup)/10)
            if n_clusters == 0:
                continue
            cluster_statistics.append([len(occup), n_clusters])

            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(dist_filter_frame[occup])
            if len(cluster_frame) == 0:
                cluster_frame = kmeans.cluster_centers_
            else:
                cluster_frame = np.vstack((cluster_frame, kmeans.cluster_centers_))
        
        new_frames.append(cluster_frame)

    frames = new_frames

for frame_num, frame in enumerate(frames):
    print(frame_num)
    box_dim = np.copy(box_dims[frame_num])

    with open('vox_output'+suffix+'.txt', 'a') as vox_file:
        vox_file.write('#voxel, center(x y z), total_area, e_bend\n')
    with open('particle_output'+suffix+'.txt', 'w') as particle_file:
        particle_file.write('#x y z area mean gauss\n')

    voxel = Voxel(frame, box_dim, voxel_dim=voxel_size)

    global_tri = []
    ## BIG LOOP over all occupied voxels
    #get points for a voxel and all neighboring voxels
    vox_output = []

    ## TESTING OPTIONS
    #index = 983 #230
    #occup = voxel.occupants[index]
    #if True:
    for index, occup in voxel.occupants.items():
        particle_output = []
        try:
            print(index, voxel.centers[index], len(occup))
        except:
            pass
        #    print("centers not defined, %i occup" % len(occup))

        # if occupants is empty
        if not occup:
            continue
        #indices of all atoms in all of the voxels around the voxel of interest
        group_ids = []
        group_xyz = np.empty((0,3))
        for neigh in voxel.neighbors[index]:
            group_ids += voxel.occupants[neigh]


            temp_xyzs = np.copy(frame[voxel.occupants[neigh],:])

            #wrap coordinates around voxel center of interest
            for temp_xyz in temp_xyzs:
                if temp_xyz[0] - voxel.centers[index][0] > ((box_dim[0,1]-box_dim[0,0])/2):
                    temp_xyz[0] -= (box_dim[0,1] - box_dim[0,0])
                if temp_xyz[0] - voxel.centers[index][0] < (-(box_dim[0,1]-box_dim[0,0])/2):
                    temp_xyz[0] += (box_dim[0,1] - box_dim[0,0])

                if temp_xyz[1] - voxel.centers[index][1] > ((box_dim[1,1]-box_dim[1,0])/2):
                    temp_xyz[1] -= (box_dim[1,1] - box_dim[1,0])
                if temp_xyz[1] - voxel.centers[index][1] < (-(box_dim[1,1]-box_dim[1,0])/2):
                    temp_xyz[1] += (box_dim[1,1] - box_dim[1,0])
            group_xyz = np.append(group_xyz, temp_xyzs,  axis=0)

        #new indices of atoms in the voxel of interest
        center_group_ids = []
        for ii, num in enumerate(group_ids):
            if (num in voxel.occupants[index]):
                center_group_ids.append(ii)

        #isomap to dimensionality reduction
        red_xy = dm.get_isomap(group_xyz)

        #triangulation in reduced coordinates
        tri = Delaunay(red_xy)

        group_array = np.array(group_ids)
        for simplex in tri.simplices:
            global_simplex = np.array( [group_array[a] for a in simplex] )
            global_tri.append(global_simplex)

        mean_curv, gauss_curv, cot_angles, neigh_distances, angle_total, p_areas = get_curvature(group_xyz, tri, subset=center_group_ids)
        for dumb in zip(group_xyz[center_group_ids], p_areas[center_group_ids], mean_curv[center_group_ids], gauss_curv[center_group_ids]):
            particle_output.append(np.hstack([dumb[0], dumb[1], dumb[2], dumb[3]]))

        ## calculate the bending energy as the sum of area*(kappa*mean_curv + kappa_g*gaussain_curv)
        kappa = 1
        kappa_g = 1
        e_bend = 0
        total_area = 0
        for ii, p in enumerate(group_xyz):
            if (p[0]>box_dim[0,0] and p[0]<box_dim[0,1] and p[1]>box_dim[1,0] and p[1]<box_dim[1,1]):
                total_area += p_areas[ii]
                e_bend += p_areas[ii] * (kappa*mean_curv[ii]**2 + kappa_g*gauss_curv[ii])
        e_bend /= total_area

        vox_output.append([index, voxel.centers[index][0], voxel.centers[index][1], voxel.centers[index][2], total_area, e_bend])
        print(vox_output[-1])

        with open('vox_output'+suffix+'.txt', 'a') as vox_file:
            vox_file.write('%i, %.4f, %.4f, %.4f, %.4f, %.8f\n' % tuple(vox_output[-1]))
        with open('particle_output'+suffix+'.txt', 'a') as particle_file:
            for dumb in particle_output:
                particle_file.write('%.4f, %.4f, %.4f, %.4e, %.4e, %.4e\n' % tuple(dumb))

