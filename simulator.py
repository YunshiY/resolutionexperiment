from __future__ import division
import matplotlib.image as mpimg
import numpy as np
import random, math
import time
import matplotlib.pyplot as plt
import scipy.stats
import random, math
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
from Bio import SeqIO,Seq
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import pytess
import networkx as nx
#import community
import os,datetime
import planarity
import ipdb
from scipy.spatial import ConvexHull
import operator
from scipy.optimize import minimize
import matplotlib
import matplotlib.cm as cm
import scipy.optimize
from itertools import izip
from scipy.optimize import OptimizeResult

def uniq(lst):
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item
def sort_and_deduplicate(l):
    return list(uniq(sorted(l,reverse=True)))

def convert_vlines_to_edges(enumerated_faces,edgelist):
    # now convert vline IDs back into edges
    edges_faces_enumerated = []
    for face in range(0, len(enumerated_faces)):
        corresponding_face = []
        for line in range(0, len(enumerated_faces[face])):
            corresponding_edge = (edgelist[enumerated_faces[face][line]][0], edgelist[enumerated_faces[face][line]][1])
            corresponding_face += [corresponding_edge]
        edges_faces_enumerated += [corresponding_face]
    return edges_faces_enumerated

def get_vline_size_of_face(face_entry, all_vlines):
    vsizes_enumerated_faces = []
    for vline in range(0,len(face_entry)):
        vline_size = all_vlines[face_entry[vline]][2] - all_vlines[face_entry[vline]][1]
        vsizes_enumerated_faces += [vline_size]
    return vsizes_enumerated_faces

def sort_face_by_xposn(facetosort,all_vlines):
    x_positions = np.zeros(len(facetosort))
    for edge_to_sort in range(0, len(facetosort)):
        x_positions[edge_to_sort] = all_vlines[facetosort[edge_to_sort]][0]
    sorted_face = [x for _, x in sorted(zip(x_positions, facetosort))]
    return sorted_face
def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]
def scale_image_axes(points, filename = 'monalisacolor.png'):
    imarray = np.array(mpimg.imread(filename))
    x_length_image = len(imarray[0,:])
    y_length_image = len(imarray[:,0])
    x_grid_min = min(points[:,0])
    y_grid_min = min(points[:, 1])
    x_grid_max = max(points[:,0])
    y_grid_max = max(points[:,1])
    x_length_grid = x_grid_max-x_grid_min
    y_length_grid = y_grid_max-y_grid_min
    aspect_ratio_image = x_length_image/y_length_image
    aspect_ratio_grid = x_length_grid/y_length_grid
    if aspect_ratio_image <= aspect_ratio_grid: #then this means that the image is tall relative to the grid
        # in this case, we add padding to the sides of the image while maintaining the original height
        padding_one_side = np.ones((len(imarray[:,0]),int((1./aspect_ratio_grid*len(imarray[:,0])-len(imarray[0,:]))/2) ,np.shape(imarray)[2]))
        try :scaled_imarray = np.concatenate((padding_one_side,imarray),axis=1)
        except: ipdb.set_trace()
        scaled_imarray = np.concatenate((scaled_imarray, padding_one_side), axis=1)/np.max(imarray)
    elif aspect_ratio_image > aspect_ratio_grid:
        # in this case, we add padding to the top and bottom with same width as the original image

        padding_one_side = np.ones(( int((aspect_ratio_grid*len(imarray[0,:]) - len(imarray[:,0]))/2) , len(imarray[0,:]) , 3))
        scaled_imarray = np.concatenate((padding_one_side, imarray), axis=0)
        scaled_imarray = np.concatenate((scaled_imarray, padding_one_side), axis=0) / np.max(imarray)
    else:
        print 'error with aspect ratio scaling of image'
    lookup_x_axis = np.arange(x_grid_min,x_grid_max,x_length_grid/float(len(scaled_imarray[:,0])-1))
    lookup_y_axis = np.arange(y_grid_min,y_grid_max,y_length_grid/float(len(scaled_imarray[0,:])-1))
    return scaled_imarray, lookup_x_axis, lookup_y_axis

def get_pixel_rgb(scaled_imarray, lookup_x_axis, lookup_y_axis, grid_x_lookup_value,grid_y_lookup_value):
    xindex = np.where(lookup_x_axis==find_nearest(lookup_x_axis,grid_x_lookup_value))[0][0]
    yindex = np.where(lookup_y_axis==find_nearest(lookup_y_axis,grid_y_lookup_value))[0][0]
    rgb = scaled_imarray[xindex,yindex]
    gray = np.average(scaled_imarray[xindex,yindex])
    return rgb, gray

def update_enumerated_faces(enumerated_faces, vsizes_enumerated_faces, partial_face,partial_face_vsize, all_vlines, edgelist):
    face_added = False
    # print 'CANDIDATE VLINE IDS:', partial_face, 'CANDIDATE SIZES: ', partial_face_vsize,'candidate edges: ', edgelist[partial_face[0]][0:2], ' and ', edgelist[partial_face[1]][0:2]
    # if partial_face == [0,7] or partial_face == [0,4]:
    #     ipdb.set_trace()
    if len(enumerated_faces) == 0:
        enumerated_faces += [partial_face]
        vsizes_enumerated_faces += [get_vline_size_of_face(enumerated_faces[0], all_vlines)]
        face_added = True
        # print 'AN EMPTY ENUMERATED LIST WAS FOUND'
        print 'FACE ADDED via empty start: : : : : ', partial_face, ' to ', enumerated_faces,'candidate edges: ', edgelist[partial_face[0]][0:2], ' and ', edgelist[partial_face[1]][0:2]
    else:
        for face in range(0, len(enumerated_faces)):
            # ipdb.set_trace()
            # print 'FACE# ', face, 'vlines logged so far: ', enumerated_faces[face], 'and their sizes: ', vsizes_enumerated_faces[face]
            # above we sorted based on spatial order, face to the left or right of the vline in question
            # we organize the partially completed faces so that they are also spatially ordered
            # what we want to know is which side of the face is incomplete
            # every face has a large edge on one side, and multiple smaller edges on the other side
            # to merge a partial face with one of the existing partial faces, we must first rule out any that have ...
            # a different largest side
            # then we add only those vlines which are of the smaller side
            ####   check if the two partial faces have the same size symmetry ####
            if (vsizes_enumerated_faces[face][0] - vsizes_enumerated_faces[face][-1]) * (partial_face_vsize[0] - partial_face_vsize[1]) > 0:  # then they are of same sign
                # print 'compatible symetries. size of enum first element, last, candidate first, last: ',vsizes_enumerated_faces[face], partial_face_vsize
                # next we need to find out if the two faces are not just compatible in terms of size, but also if their largest
                # edge is in fact the same edge. these two criteria, i.e. having the same largest edge and that it be on the same left/right side is sufficient to say
                # that they are of the same face! some may have same largest edge but be on two different sides of that edge for example - being different faces
                # we can use argmax of both partial faces in order to ignore having to check which side it's on
                if enumerated_faces[face][np.argmax(vsizes_enumerated_faces[face])] == partial_face[np.argmax(partial_face_vsize)]:
                    # print 'largest edge match (enumed, candidate): ',enumerated_faces[face], partial_face
                    # now we have determined that the two faces are the same, we must then merge them except in the case where
                    # they are not just the same face, but the same set of edges - a redundant entry, which we can then skip
                    smaller_edge = partial_face[np.argmin(partial_face_vsize)]
                    if smaller_edge not in enumerated_faces[face]:
                        # print 'non-redundancy criterion ok (smaller edge of candidate, enumed face:', smaller_edge, enumerated_faces[face]
                        # now that we know that the two faces are same and that the entries are different, we can perform a merge action
                        # next use argmin of the partial face in question to send the un-added edge into the incomplete enumerated face
                        # while we're at it, we should sort the final entry by xpos
                        ###MERGE MERGE MERGE ###

                        ### SORT SORT SORT ###
                        ##### check if this is an outer face first, if so then shift leftmost edge to the other end

                        sorted_face = sort_face_by_xposn(facetosort=enumerated_faces[face],
                                    all_vlines=all_vlines)
                        sorted_after_merging = sort_face_by_xposn(facetosort=enumerated_faces[face]+[smaller_edge],
                                    all_vlines=all_vlines)
                        # if len(enumerated_faces[face])>2:
                        first_vlines_xposn = all_vlines[enumerated_faces[face][0]][0]
                        second_vlines_xposn = all_vlines[enumerated_faces[face][1]][0]
                        last_vlines_xposn = all_vlines[enumerated_faces[face][-1]][0]
                        secondtolast_vlines_xposn = all_vlines[enumerated_faces[face][-2]][0]
                        if second_vlines_xposn < first_vlines_xposn: #then this is an outer face, and after sorting we must shift stack
                            enumerated_faces[face] = [sorted_after_merging[-1]] + sorted_after_merging[0:-1]
                        elif last_vlines_xposn < secondtolast_vlines_xposn: #then this is the other type of outer face
                            enumerated_faces[face] = sorted_after_merging[1:]+ [sorted_after_merging[0]]
                        elif first_vlines_xposn < second_vlines_xposn and secondtolast_vlines_xposn < last_vlines_xposn:
                            #then the array appears to be sorted normally, meaning it is a face that does not span the periodic boundary
                            enumerated_faces[face] = sorted_after_merging
                        else:
                            print 'error with face sorting during edge merge attempt'
                            ipdb.set_trace()
                        # else:
                        #     enumerated_faces[face] = sorted_after_merging



                        vsizes_enumerated_faces[face] = get_vline_size_of_face(enumerated_faces[face],all_vlines)
                        face_added = True
                        print 'FACE ADDED via merge event : : : : : ', partial_face, ' to ', enumerated_faces[face], 'edges are ', convert_vlines_to_edges(enumerated_faces, edgelist)[face],'candidate edges: ', edgelist[partial_face[0]], ' and ', edgelist[partial_face[1]]
                        break
                    else:
                        # print 'REJECT we found a redundant face: ', enumerated_faces[face], ' and ', partial_face
                        # this must be a redundant face, we should not readd it to the archive
                        face_added = True
                        break
                else:
                    pass# print 'REJECT - despite having same spatial class, the largest edge was not the same edge for the two faces: ', enumerated_faces[face][np.argmax(vsizes_enumerated_faces[face])], ' and ',  partial_face[np.argmax(partial_face_vsize)]
            else:
                pass # print 'REJECT - incompatible symmetries: ', vsizes_enumerated_faces[face], ' vs ', partial_face_vsize
    # now that we've checked for this very special case of a face which shares both same largest edge and same
    # left-right symmetry of the largest edge and finally accounted for redundant entries and merged such a case
    # if we made it through without triggering either of the face-added status
    # then we can simply add the partial entry as a new incomplete face to the list of enumerated faces
    # it will then be revisited for additions until completed
    if face_added == False:
        enumerated_faces += [partial_face]
        vsizes_enumerated_faces += [get_vline_size_of_face(enumerated_faces[-1], all_vlines)]
        face_added = True
        print 'FACE ADDED via last resort : : : : : ', partial_face, ' to ', enumerated_faces,'candidate edges added: ', edgelist[partial_face[0]][0:2], ' and ', edgelist[partial_face[1]][0:2]
    face_edge_list = convert_vlines_to_edges(enumerated_faces, edgelist)
    # ipdb.set_trace()
    return enumerated_faces, vsizes_enumerated_faces

def planar_embedding_draw(graph, labels=True, title=''):
    """Draw planar graph with Matplotlib."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        from matplotlib.collections import PatchCollection
    except ImportError:
        raise ImportError("Matplotlib is required for draw()")
    pgraph = planarity.PGraph(graph)
    pgraph.embed_drawplanar()
    hgraph = planarity.networkx_graph(pgraph) #returns a networkx graph built from planar graph- nodes and edges only
    patches = []
    node_labels = {}
    xs = []
    ys = []
    all_vlines = []
    xmax=0
    for node, data in hgraph.nodes(data=True):
        y = data['pos']
        xb = data['start']
        xe = data['end']
        x = int((xe+xb)/2)
        node_labels[node] = (x, y)
        patches += [Circle((x, y), 0.25)]#,0.5,fc='w')]
        xs.extend([xb, xe])
        ys.append(y)
        plt.hlines([y], [xb], [xe])
    edgelist = list(hgraph.edges(data=True)) #only 

    for i in range(0,len(edgelist)):
        #print i
        data = edgelist[i][2]
        x = data['pos']
        if x > xmax: xmax = x
        yb = data['start']
        ye = data['end']
        ys.extend([yb, ye])
        xs.append(x)
        all_vlines += [[x, yb,ye]]
        plt.vlines([x], [yb], [ye])
    # labels
    if labels:
        for n, (x, y) in node_labels.items():
            plt.text(x, y, n,
                     horizontalalignment='center',
                     verticalalignment='center',
                     bbox = dict(boxstyle='round',
                                 ec=(0.0, 0.0, 0.0),
                                 fc=(1.0, 1.0, 1.0),
                                 )
                     )
    p = PatchCollection(patches)
    ax = plt.gca()
    ax.add_collection(p)
    plt.axis('equal')
    plt.xlim(min(xs)-1, max(xs)+1)
    plt.ylim(min(ys)-1, max(ys)+1)
    plt.savefig(directory_name + '/' +'planar_embedding_diagram'+title+'.svg')
    plt.savefig(directory_name + '/' + 'planar_embedding_diagram' + title + '.png')
    plt.close()
    xmax += 1
    enumerated_faces = []
    vsizes_enumerated_faces = []
    for vline in range(0,len(all_vlines)):
        print vline
        #find closest interesting vline for top part, and bottom part resp.
        minimum_positive_distance = 999999999999
        edge_candidate_positive = []
        minimum_negative_distance = -999999999999
        edge_candidate_negative = []
        edge_candidate_primary = [edgelist[vline][0],edgelist[vline][1],edgelist[vline][2]['pos']]
        #intersectability - we create an intersectors group,
        #ie the set of vlines that have vertical overlap with our current vline
        #start by making an empty intersectors list that we will add to below.
        # ultimately all we want from these are the minima, the vlines closest to our vline with the potential to ...
        # ...intersect, these will be the starts of faces
        #note that each intersector has two associated lateral distances, from left and right, negative and positive
        #this means we enforce a periodic boundary condition, so in the case of othervline being to the right of vline,
        #then we would add the distance from the leftmost edge to vline with the distance from rightmost edge to other_vline
        intersectors = [[],[],[]] #[[identifying index of other_vline],[distance btw vline and other_vline to the left],[" for distance via the right]]
        # ipdb.set_trace()
        for other_vline in range(0, len(all_vlines)):
            #check vertical intersectability
            if all_vlines[vline][1] >= all_vlines[other_vline][2] or all_vlines[vline][2] <= all_vlines[other_vline][1] or all_vlines[vline][0] == all_vlines[other_vline][0]:
                pass
            else: #if vertical overlap exists, then we need to add the other_vline to the list of intersectors of vline
                intersectors[0] += [other_vline]
                distance_to_right = (all_vlines[vline][0] - all_vlines[other_vline][0])%xmax
                distance_to_left = (all_vlines[other_vline][0] - all_vlines[vline][0])%xmax
                intersectors[2] += [distance_to_left]
                intersectors[1] += [distance_to_right]
        closest_to_left = intersectors[0][np.argmin(intersectors[1])]
        partial_face_to_left = [closest_to_left,vline]
        partial_face_to_left_vsize = [all_vlines[closest_to_left][2]-all_vlines[closest_to_left][1],all_vlines[vline][2]-all_vlines[vline][1]]
        closest_to_right = intersectors[0][np.argmin(intersectors[2])]
        partial_face_to_right = [vline,closest_to_right]
        partial_face_to_right_vsize = [all_vlines[vline][2]-all_vlines[vline][1],all_vlines[closest_to_right][2]-all_vlines[closest_to_right][1]]
        enumerated_faces,vsizes_enumerated_faces = update_enumerated_faces(enumerated_faces, vsizes_enumerated_faces, partial_face_to_left, partial_face_to_left_vsize, all_vlines, edgelist)
        enumerated_faces,vsizes_enumerated_faces = update_enumerated_faces(enumerated_faces, vsizes_enumerated_faces, partial_face_to_right, partial_face_to_right_vsize, all_vlines,edgelist)
        face_edge_list = convert_vlines_to_edges(enumerated_faces,edgelist)
        # print        "END OF common face insertion block"
    # print 'END of vline loop'
    print 'END OF FUNCTION END OF FUNCTION END OF FUNCTION END OF FUNCTION END OF FUNCTION END OF FUNCTION '
    return face_edge_list


# intput: a grpah as a dictionary
# output: a list of lists of vertices in the grpah which share neighbors
def same_neighbors(graph):
    same_neighbors = []
    for u in graph:
        same_neighbors_u = [u]
        for v in graph:
            if v != u:
                if set(graph[u]) == set(graph[v]):
                    same_neighbors_u.append(v)
        if len(same_neighbors_u) > 1:
            same_neighbors.append(same_neighbors_u)
    same_neighbors = [set(x) for x in same_neighbors]
    same = []
    for i in same_neighbors:
        if i not in same:
            same.append(i)
    same = [list(x) for x in same]
    return same

# input: a graph in the form of a dictionary and an outter_face in the form of a list of vertices.
def tutte_embedding(graph, outer_face):
    # ipdb.set_trace()
    pos = {}  # a dictionary of node positions
    tmp = nx.Graph()
    for edge in outer_face:
        a, b = edge
        tmp.add_edge(a, b)

    tmp_pos = circular_layout_sorted(tmp,1.0)  # ensures that outterface is a convex shape
    pos.update(tmp_pos)
    outer_vertices = tmp.nodes()
    remaining_vertices = [x for x in graph.nodes() if x not in outer_vertices]
    size = len(remaining_vertices)
    A = [[0 for i in range(size)] for i in range(
        size)]  # create the the system of equations that will determine the x and y positions of remaining vertices
    b = [0 for i in range(size)]  # the elements of theses matrices are indexed by the remaining_vertices list
    C = [[0 for i in range(size)] for i in range(size)]
    d = [0 for i in range(size)]
    for u in remaining_vertices:
        i = remaining_vertices.index(u)
        neighbors = [node for node in graph.neighbors(u)]
        n = len(neighbors)
        A[i][i] = 1
        C[i][i] = 1
        for v in neighbors:
            if v in outer_vertices:
                try: b[i] += float(pos[v][0]) / n
                except: ipdb.set_trace()
                d[i] += float(pos[v][1]) / n
            else:
                j = remaining_vertices.index(v)
                A[i][j] = -(1 / float(n))
                C[i][j] = -(1 / float(n))
    x = np.linalg.solve(A, b)
    y = np.linalg.solve(C, d)
    for u in remaining_vertices:
        i = remaining_vertices.index(u)
        pos[u] = [x[i], y[i]]
    return pos

def prefix(directory_label):
    global directory_name
    global today
    today = datetime.date.today()
    directory_name = str(today) + '_'+ str(directory_label)+ '_' + str(time.time())
    os.makedirs(directory_name)
    return directory_name

def grid(xlen,ylen,randomseed=4):
    #create a hexagonal lattice of points representing locations of oligos comprising the lawn
    np.random.seed(randomseed)
    num = xlen * ylen
    n= 2 * num
    points_1 = np.zeros((num, 2))
    points_2 = np.zeros((num, 2))
    for i in range(0, xlen):  # first grid
        for j in range(0, ylen):
            points_1[i * ylen + j, 0] = i
            points_1[i * ylen + j, 1] = (j * math.sqrt(3))
    for i in range(0, num):  # second grid
        points_2[i, 0] = points_1[i, 0] + 0.5
        points_2[i, 1] = points_1[i, 1] + math.sqrt(3) / 2
    points = np.vstack((points_1, points_2))  # paste together vertically
    np.savetxt(directory_name+'/'+'points', points, delimiter=",")
    return points#coordinates

def circular_grid_yunshi(xlen,ylen,randomseed=4):
    np.random.seed(randomseed)
    num = xlen * ylen
    n = 2 * num
    points_1 = np.zeros((num, 2))
    points_2 = np.zeros((num, 2))
    for i in range(0, xlen):  # first grid
        for j in range(0, ylen):
            points_1[i * ylen + j, 0] = i
            points_1[i * ylen + j, 1] = (j * math.sqrt(3))
    for i in range(0, num):  # second grid
        points_2[i, 0] = points_1[i, 0] + 0.5
        points_2[i, 1] = points_1[i, 1] + math.sqrt(3) / 2
    p = np.vstack((points_1, points_2))  # paste together vertically

    centerx = (p[np.argmax(p[:, 0]), 0] + p[np.argmin(p[:, 0]), 0]) / 2
    centery = (p[np.argmax(p[:, 1]), 1] + p[np.argmin(p[:, 1]), 1]) / 2

    R = 0
    if centerx <= centery:
        R = centerx
    else:
        R = centery

    counts = 0
    dis = np.zeros((n))
    for i in range(0, n):
        dis[i] = math.sqrt((p[i, 0] - centerx) ** 2 + (p[i, 1] - centery) ** 2)
        if dis[i] <= R:
            counts += 1

    points = np.zeros((counts, 2))
    count_1 = 0
    for i in range(0, n):
        if dis[i] <= R:
            points[count_1, 0] = p[i, 0]
            points[count_1, 1] = p[i, 1]
            count_1 += 1
    return points

def seed(nseed,points):
    npoints = len(points)
    choices = [i for i in range(0, npoints)]
    seed = np.random.choice(choices, nseed, replace=False)
    corseed = np.zeros((nseed, 2))
    p1 = list()
    for i in range(0, nseed):
        n = seed[i]
        corseed[i] = points[n, :]
        np.savetxt(directory_name +'/'+ 'corseed', corseed, delimiter=",")
        np.savetxt(directory_name +'/'+  'seed', seed, delimiter=",")
    def barcode():
        base = 'ATCG'
        barcd = ''
        for i in range(0, 5):
            barcd += random.choice(base)
        return barcd

    bc = ["" for x in range(0, nseed)]
    for i in range(0, nseed):

        bc[i] = barcode()
        if i >= 0:
            for j in range(0, i - 1):
                while bc[i] == bc[j]:
                    bc[i] = barcode()

    np.savetxt(directory_name+'/'+ 'barcode', bc, delimiter=",", fmt="%s")
    bc_record = ["" for i in bc]
    for i in range(0, len(bc)):
        bc_record[i] = SeqRecord(Seq(bc[i]), id=str(i))
    SeqIO.write(bc_record, directory_name+'/'+ "bc.faa", "fasta")
    for i in range(0, nseed):
        p1.append(str(bc_record[i].seq.reverse_complement()))
    np.savetxt(directory_name+'/'+ 'p1', p1, delimiter=',', fmt='%s')
    return corseed,p1,bc,seed#p1,bc are sequence,seed is index

def convert_dict_to_list(dict):
    dictList = []
    temp = []
    for key, value in dict.iteritems():
        temp = [value[0],value[1]]
        dictList.append(temp)
    return dictList

def rotate(x,y,xo,yo,theta): #rotate x,y around xo,yo by theta (rad)
    xr=math.cos(theta)*(x-xo)-math.sin(theta)*(y-yo)   + xo
    yr=math.sin(theta)*(x-xo)+math.cos(theta)*(y-yo)  + yo
    return [xr,yr]

def rotate_series(series,theta,originx,originy):
    rotated_series = np.zeros((len(series),2))
    for coordinate in range(0,len(series)):
        new_coordinate = rotate(series[coordinate][0], series[coordinate][1], originx, originy, theta)
        # ipdb.set_trace()
        rotated_series[coordinate][0] = new_coordinate[0]
        rotated_series[coordinate][1] = new_coordinate[1]
    return rotated_series

def align(corseed,reconstructed_points,reconstructed_graph='None',title='alignment',error_threshold = 50,edge_number = 0):
    minerror = 99999999
    if edge_number == 0:
        edge_number = len(corseed)
    minevals = 0
    while minerror > error_threshold and minevals < 1000:
        corseed = corseed * np.random.choice([-1.,1.],2) #generates a 2 item list of either -1s or +1s to flop coordinates
        initialize = np.array([random.uniform(0.,6.3),random.uniform(-10, 10.),random.uniform(-10, 10.),random.uniform(0.,1)])
        res1 = minimize(fun=evaluate_distortion, x0=initialize,#x0=np.array([.1, 6., 2., .9]),
                         args=([corseed, reconstructed_points, False,edge_number],), method='TNC', jac=None,
                         bounds=[(0., float('inf')),
                                 (float('-inf'), float('inf')),
                                 (float('-inf'), float('inf')),
                                 (.001, float('inf'))],
                        tol=1e-12,options={'eps': 1e-08,
                                           'scale': None,
                                           'offset': None,
                                           'mesg_num': None,
                                           'maxCGit': -1,
                                           'maxiter': 1000000,
                                           'eta': -1,
                                           'stepmx': 0,
                                           'accuracy': 0,
                                           'minfev': 1000,
                                           'ftol': -1,
                                           'xtol': -1,
                                           'gtol': -1,
                                           'rescale': -1,
                                           'disp': False}
                    )
        # alignment_minimizer_kwargs = {"tol":0.01,'method':'TNC',"args": ([corseed, reconstructed_points, False, edge_number],)}
        # res1 = scipy.optimize.basinhopping(func=evaluate_distortion,
        #                             x0=np.array([.1, 6., 2., .9]),
        #                             minimizer_kwargs=alignment_minimizer_kwargs,
        #                             niter=1000000,
        #                             disp=False,
        #                             niter_success=200,
        #                             callback=callback_on_optimization_indicator)
        if minerror > res1['fun']:
            minerror = res1['fun']
        else:
            pass
        minevals += 1

    xopt1 = res1['x']
    error1, corseed_adjusted = evaluate_distortion(xopt1,args=[corseed, reconstructed_points,True,True,edge_number])
    plt.close()
    plt.scatter(corseed_adjusted[:,0],corseed_adjusted[:,1])
    plt.scatter(reconstructed_points[:,0],reconstructed_points[:,1])
    distances = []
    for i in range(0, len(reconstructed_points)):
        plt.text(corseed_adjusted[i, 0], corseed_adjusted[i, 1], '%s' % i, alpha=0.5)
        plt.text(reconstructed_points[i, 0], reconstructed_points[i, 1], '%s' % i, alpha=0.5)
        plt.plot([corseed_adjusted[i, 0],reconstructed_points[i, 0]],[corseed_adjusted[i, 1],reconstructed_points[i, 1]],'k-')
        distances += [np.sqrt((corseed_adjusted[i, 0]-reconstructed_points[i, 0])**2+(corseed_adjusted[i, 1]-reconstructed_points[i, 1])**2)]
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.xlim(-1.1,1.1)
    plt.ylim(-1.1, 1.1)
    tri1 = Delaunay(corseed_adjusted)
    plt.triplot(corseed_adjusted[:, 0], corseed_adjusted[:, 1], tri1.simplices.copy(),'b-',alpha=0.2,marker='')
    if reconstructed_graph == 'None':
        tri2 = Delaunay(reconstructed_points)
        plt.triplot(reconstructed_points[:, 0], reconstructed_points[:, 1], tri2.simplices.copy(),'r-',alpha=0.2,marker='')
    else:
        nx.draw_networkx_edges(reconstructed_graph,reconstructed_points,width=1.0,alpha=0.2,edge_color='r')
        nx.draw_networkx_nodes(reconstructed_graph, reconstructed_points,node_color='r',node_size=25,alpha=0.2)
    plt.savefig(directory_name+'/'+ title +'redblue.svg')
    plt.savefig(directory_name + '/' + title + 'redblue.png')
    plt.close()

    cmap = matplotlib.cm.get_cmap('YlOrRd')
    normalize = matplotlib.colors.Normalize(vmin=min(distances), vmax=max(distances))
    colors = [cmap(normalize(value)) for value in distances]

    plt.scatter(corseed_adjusted[:,0],corseed_adjusted[:,1],c='k', alpha=0.1)
    plt.scatter(reconstructed_points[:,0],reconstructed_points[:,1],c='k',marker='o', alpha=0.1)
    for i in range(0, len(reconstructed_points)):
        plt.plot([corseed_adjusted[i, 0],reconstructed_points[i, 0]],[corseed_adjusted[i, 1],reconstructed_points[i, 1]],'-',color=colors[i])
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.xlim(-1.1,1.1)
    plt.ylim(-1.1, 1.1)
    plt.savefig(directory_name+'/'+ title +'heatmap.svg')
    plt.savefig(directory_name + '/' + title + 'heatmap.png')
    print 'final cost function value: ',minerror
    return distances

def get_radial_profile(points,distances,title='radial_error_plot'):
    plt.close()
    binno = len(points)/10.
    ideal_radii = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    bins = np.arange(0.0,1.0,1.0/binno)
    binned_sums = np.zeros((len(bins)))
    binned_counts = np.zeros((len(bins)))
    for bin in range(0,len(points)):
        binned_sums[searchify(bins,ideal_radii[bin])] += distances[bin]
        binned_counts[searchify(bins, ideal_radii[bin])] += 1
    binned_averages = binned_sums/binned_counts
    plt.scatter(ideal_radii,distances,c='k',marker='o', alpha=0.5)
    plt.plot(bins, binned_averages,'r-')
    plt.xlabel('normalized radius')
    plt.ylabel('normalized polony displacement')
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.ylim(0,1.0)
    plt.savefig(directory_name+'/'+title+'.svg')
    plt.savefig(directory_name + '/' + title + '.png')
    plt.close()

def searchify(array, value):
    array = np.asarray(array)
    i = (np.abs(array - value)).argmin()
    return i

def evaluate_distortion(parameters, args):
    theta = parameters[0]
    x_translate = parameters[1]
    y_translate = parameters[2]
    scale = parameters[3]
    corseed, reconstructed, full_output,edge_number = args[0], args[1], args[2],args[3]
    master_error = 0
    #scale uniformly
    corseed = corseed*scale
    #translate all points to a new origin
    corseed[:,0] = corseed[:,0] + x_translate
    corseed[:, 1] = corseed[:, 1] + y_translate
    #rotate the reconstructed about the origin 0,0 since we have arrayed around unit circle
    corseed = rotate_series(corseed, theta,x_translate,y_translate)
    all_distances = np.zeros((len(corseed)))
    for polony in range(0,len(corseed)):
        distance = np.sqrt((corseed[polony][0]-reconstructed[polony][0])**2+(corseed[polony][1]-reconstructed[polony][1])**2)
        all_distances[polony] = distance
    mean_distance = np.average(all_distances)
    mean_cubed_distance = np.average(all_distances**3)
    stdev_distance = np.std(all_distances)
    skewness_distance = scipy.stats.skew(all_distances)
    master_error = (mean_cubed_distance*np.sqrt(len(corseed)))#*(np.abs(skewness_distance)+1) #this means a perfectly symmetric distance distribution would not alter the mean distance
    if full_output == False:
        return master_error
    else:
        return master_error,corseed

def rca(p1,points,seed,scaled_imarray, lookup_x_axis, lookup_y_axis):
    npoints = len(points)
    nseed = len(p1)
    p2_proxm = np.zeros((npoints))
    p2_secondary = np.zeros((npoints))

    def reverse(seq):
        base = 'ATCG'
        rbase = 'TAGC'
        n = len(seq)
        rseq = ""
        for i in range(0, n):
            for j in range(0, 4):
                if seq[n - i - 1] == base[j]:
                    rseq += rbase[j]
        return rseq



    for i in range(0, npoints):
        p2_dis = np.zeros((nseed))
        for j in range(0, nseed):
            p2_dis[j] = (math.sqrt((points[i, 0] - seed[j, 0]) ** 2 + (points[i, 1] - seed[j, 1]) ** 2))
        p2_proxm[i] = np.argsort(p2_dis)[0]
        p2_secondary[i] = np.argsort(p2_dis)[1]


    p2 = ["" for x in range(0, npoints)]
    for i in range(0, npoints):
        x = int(p2_proxm[i])
        p2[i] = reverse(p1[x])

    bc_index_pair = zip(p2_proxm, p2_secondary)
    np.savetxt(directory_name+'/'+ 'p2', p2, delimiter=",", fmt="%s")
    np.savetxt(directory_name+'/'+ 'polony_seed', p2_proxm, delimiter=",")
    np.savetxt(directory_name+'/'+ 'point_adjacent_polony', p2_secondary, delimiter=",")
    np.savetxt(directory_name+'/'+ 'cheating_sheet', bc_index_pair, delimiter=',')
    color_cycle = ['orange', 'green', 'blue', 'pink', 'cyan', 'gray']




    vor_polony = Voronoi(seed, qhull_options = "")
    voronoi_plot_2d(vor_polony,show_vertices=False)
    rgb_stamp_catalog = np.zeros((nseed,3))
    for i in range(0, nseed):
        c = c = i % 6
        for j in range(0, npoints):
            if p2_proxm[j] == i:
                plt.scatter(points[j, 0], points[j, 1], color=color_cycle[c], s=1)
                rgb, gray = get_pixel_rgb(scaled_imarray,lookup_x_axis,lookup_y_axis,points[j,0],points[j,1])
                rgb = rgb / np.sum(rgb)
                p_threshold = random.random()
                if p_threshold < gray: #darker means closer to zero,
                    try:
                        colorID = np.random.choice([0,1,2],p=rgb)
                        rgb_stamp_catalog[i][colorID] += 1
                    except:
                        rgb_stamp_catalog[i][0] += 1
                        rgb_stamp_catalog[i][1] += 1
                        rgb_stamp_catalog[i][2] += 1

    plt.rcParams['figure.figsize'] = (20,20)
    if nseed < 100:
        for i in range(0, nseed):
            plt.text(seed[i, 0], seed[i, 1], '%s' % i, alpha=0.5)
    plt.xlim(-.1*(np.max(points[:,0])-np.min(points[:,0]))+np.min(points[:,0]),.1*(np.max(points[:,0])-np.min(points[:,0]))+np.max(points[:,0]))
    plt.ylim(-.1*(np.max(points[:,1])-np.min(points[:,1]))+np.min(points[:,1]), .1*(np.max(points[:,1])-np.min(points[:,1]))+np.max(points[:, 1]))
    plt.savefig(directory_name+'/'+'polony.svg')
    plt.savefig(directory_name + '/' + 'polony.png')
    plt.close()
    print 'p2 is sequence, p2_secd,p2_prim are indexes, cheating sheet saves index hybrid'

    tri = Delaunay(seed, qhull_options = "Qz Q2")
    plt.triplot(seed[:,0],seed[:,1],tri.simplices.copy())
    plt.plot(seed[:,0],seed[:,1],'o')
    plt.rcParams['figure.figsize'] = (20, 20)
    for i in range(0, nseed):
        plt.text(seed[i, 0], seed[i, 1], '%s' % i, alpha=0.5)
    plt.savefig(directory_name + '/' +'delaunay_polony.svg')
    plt.savefig(directory_name + '/' + 'delaunay_polony.png')
    plt.close()
    # ipdb.set_trace()
    return p2,p2_secondary,p2_proxm,bc_index_pair, tri,rgb_stamp_catalog

def perfect_hybrid(p2,p2_secondary,bc):
    npoints=len(p2)
    perfect_hybrid = ["" for x in range(0, npoints)]
    for i in range(0, npoints):
        perfect_hybrid[i] = p2[i] + bc[int(p2_secondary[i])]
    np.savetxt(directory_name+'/'+ 'perfect_hybrid', perfect_hybrid, delimiter=",", fmt="%s")
    print 'hybrid is sequence  '
    return perfect_hybrid

def hybrid(p1,p2,points,seed):
    npoints = len(points)
    np1 = len(p1)

    def reverse(seq):
        base = 'ATCG'
        rbase = 'TAGC'
        n = len(seq)
        rseq = ""
        for i in range(0, n):
            for j in range(0, 4):
                if seq[n - i - 1] == base[j]:
                    rseq += rbase[j]
        return rseq

    def weighted_choice(weights):
        totals = []
        running_total = 0
        for w in weights:
            running_total += w
            totals.append(running_total)
        rnd = random.random() * running_total
        for i, total in enumerate(totals):
            if rnd < total:
                return i
        region = np.zeros((1))
        int_dis = np.zeros((npoints, npoints))
        p = np.zeros((npoints))
        delta = 1
        s = 0
        r = 5
        for i in range(0, npoints - 1):  # points-points distance
            for j in range(i + 1, npoints):
                if int_dis[i, j] == 0:
                    int_dis[i, j] = (
                        math.sqrt((points[i, 0] - points[j, 0]) ** 2 + (points[i, 1] - points[j, 1]) ** 2))
                    int_dis[j, i] = int_dis[i, j]
        end = time.time()


        start = time.time()
        from scipy.stats import norm, multivariate_normal
        hybrid = ["" for x in range(0, npoints)]
        trash = np.zeros((npoints))
        for i in range(0, npoints):
            if trash[i] == 0:
                mul_nor = multivariate_normal(mean=points[i], cov=[[delta, s], [s, delta]])
                for j in range(i + 1, npoints):
                    if 0 < int_dis[i, j] <= r and trash[j] == 0:
                        p[j] = mul_nor.pdf(points[j])
                x = weighted_choice(p)
                p = np.zeros((npoints))
                if x != None:
                    hybrid[i] = p2[i] + p2[x]
                    trash[x] = 1
                    hybrid[x] = p2[x] + p2[i]
                    plt.plot([points[i, 0], points[x, 0]], [points[i, 1], points[x, 1]])
                else:
                    hybrid[i] = ''




        np.savetxt(directory_name+'/'+ 'hybrid', hybrid, delimiter=",",fmt="%s")  # loadtxt will exclude empty strings automatically

def cell(points,ncell,seed,hybrid,xlen,ylen):
    npoints = len(points)
    nseed = len(seed)
    color_cycle = ['orange', 'green', 'blue', 'pink', 'cyan', 'gray']
    corcell = np.zeros((ncell, 2))
    choices = [i for i in range(0, npoints)]
    cell = np.random.choice(choices, ncell, replace=False)
    for i in range(0, ncell):
        n = cell[i]
        corcell[i] = points[n, :]

    np.savetxt(directory_name+'/'+ 'cell_cordinate', corcell, delimiter=',')
    voronoi = pytess.voronoi(corcell)
    voronoipolys = list(voronoi)


    for i in range(0, len(voronoipolys)):
        for j in range(0, len(voronoipolys[i][1])):

            voronoipolys[i][1][j] = list(voronoipolys[i][1][j])
            if voronoipolys[i][1][j][0] < 0:#set x coordinates lower limit
                voronoipolys[i][1][j][0] = 0
            if voronoipolys[i][1][j][0] > xlen:#set x coord upper limit
                voronoipolys[i][1][j][0] = xlen
            if voronoipolys[i][1][j][1] > ylen*math.sqrt(3):#set y
                voronoipolys[i][1][j][1] = ylen*math.sqrt(3)

            if voronoipolys[i][1][j][1] < 0:
                voronoipolys[i][1][j][1] = 0

    for i in range(0, ncell):
        plt.text(corcell[i, 0], corcell[i, 1], '%s' % i, alpha=1, fontweight='bold')


    for i in range(0, len(voronoipolys)):
        plt.fill(*zip(*voronoipolys[i][1]), alpha=0.2)
    plt.savefig(directory_name+'/'+'cell.svg')
    vor_polony = Voronoi(seed)
    voronoi_plot_2d(vor_polony)
    for i in range(0, nseed):
        plt.text(seed[i, 0], seed[i, 1], '%s' % i, alpha=0.5)


    plt.savefig(directory_name+'/'+ 'cell_mask.svg')
    near = np.zeros((npoints))
    dis = np.zeros((ncell))

    group_cell = list()
    group_receptor = list()
    for i in range(0, npoints):
        for j in range(0, ncell):
            dis[j] = (math.sqrt((points[i, 0] - corcell[j, 0]) ** 2 + (points[i, 1] - corcell[j, 1]) ** 2))
        near[i] = np.argmin(dis)
    for i in range(0, ncell):
        subgroup = list()
        for j in range(0, npoints):
            if near[j] == i:
                subgroup.append(j)
        group_cell.append(subgroup)
    np.save(directory_name+'/'+ 'point_cell', group_cell)
    cell_barcode = list()
    for i in range(0, ncell):
        single_cell_barcode = list()
        for j in group_cell[i]:

                single_cell_barcode.append(hybrid[j])
        cell_barcode.append(single_cell_barcode)



    np.savetxt(directory_name+'/'+ 'cell_barcode', cell_barcode, delimiter=",", fmt="%s")
    np.save(directory_name+'/'+ 'cell_bc', cell_barcode)
    print'cell_points save  index of points belong to that cell'
    return group_cell

def get_ideal_graph(delaunay):
    G = nx.Graph()
    G.add_edges_from(np.column_stack((delaunay.simplices[:,0],delaunay.simplices[:,1])))
    G.add_edges_from(np.column_stack((delaunay.simplices[:,1],delaunay.simplices[:,-1])))
    G.add_edges_from(np.column_stack((delaunay.simplices[:,-1],delaunay.simplices[:,0])))
    return G

###########complete face traingulation ###########################

def edge_to_node(face_edge):
    face_vertices = list()
    for i in face_edge:
        face_vertices.append(i[0])
        face_vertices.append(i[1])
    face_vertices = set(face_vertices)
    return face_vertices
def centroid(face_node,node_pos):

    face_node_pos=[node_pos[i] for i in face_node]
    x_list=[vertex[0] for vertex in face_node_pos]
    y_list=[vertex[1] for vertex in face_node_pos]
    nvertex=len(face_node)
    x=sum(x_list)/nvertex
    y=sum(y_list)/nvertex
    return x,y


def vertex_CCW_sort(face_edge_list,node_pos): ##sort node incident to each face and sort face by number of nodes(decreasing)
    face_node_list=[]
    nnode=[]
    for i in face_edge_list:
        face_node=edge_to_node(i)
        face_node=list(face_node)
        nnode.append(len(face_node))
        x_centroid,y_centroid=centroid(face_node,node_pos)
        vector_list=list()

        for j in face_node:
            vector_list.append(((node_pos[j][1]-y_centroid),(node_pos[j][0]-x_centroid)))
        arcangle=[np.arctan2(vector[0],vector[1]) for vector in vector_list]
        idx=np.argsort(arcangle)
        new=[face_node[sorted] for sorted  in idx]
        face_node_list.append(new)
    outer_face=np.argmax(nnode)
    face_node_list.remove(face_node_list[outer_face])

    return face_node_list

def triangulation_fix(face_edge_list,node_pos):
    face_node_list=vertex_CCW_sort(face_edge_list,node_pos)
    new_edge=[]
    while len(face_node_list)>0:
        check=face_node_list.pop()
        if len(check)>3:
            new_edge.append((check[0],check[2]))
            face_node_list.insert(0,check[0:3])
            check.remove(check[1])
            face_node_list.insert(0, check)
        if face_node_list==None:
            break
    return new_edge


###########################################
def cheating_alignment(cheating_sheet,bc,group_cell,ideal_graph,rgb_stamp_catalog,points,nseed):
    ajc_polony = np.zeros((len(bc), len(bc)))
    ajc_polony.astype(int)
    for i in range(0, len(cheating_sheet)):
        r = int(cheating_sheet[i][0])
        c = int(cheating_sheet[i][1])
        ajc_polony[r, c] = 1
        ajc_polony[c, r] = 1

    np.savetxt(directory_name+'/'+ 'ajc_polony', ajc_polony, delimiter=",", fmt="%i")
    cell_hybrid = list()
    for i in range(0, len(group_cell)):
        subgroup = list()
        for j in group_cell[i]:  ######it's easy to get lost in index and value as index
            subgroup.append(cheating_sheet[j])
        cell_hybrid.append(subgroup)

    cell_polony = list()
    for i in range(0, len(cell_hybrid)):
        subgroup = list()
        for j in cell_hybrid[i]:
            for k in j:
                subgroup.append(j[0])

        subgroup = np.unique(subgroup)
        subgroup = subgroup[~np.isnan(subgroup)]
        cell_polony.append(subgroup)
    polony_cell = np.zeros(len(bc))
    for i in range(0, len(cell_polony)):
        for j in cell_polony[i]:
            polony_cell[int(j)] = i



    All = nx.Graph()
    for i in range(0, ajc_polony.shape[0]):
        All.add_node(i)
        for j in range(0, ajc_polony.shape[1]):
            if ajc_polony[i][j] == 1:
                All.add_edge(i, j)

    for i in range(0, len(cell_polony)):
        for j in cell_polony[i]:
            All.node[int(j)]['cell'] = i


    plt.close()

    #check for planarity
    # metrics are removed however
    if (planarity.is_planar(ideal_graph))==True and nx.is_connected(ideal_graph)==True:
        ideal_faces = planar_embedding_draw(ideal_graph,title='ideal')
        plt.savefig(directory_name + '/' + 'ideal_planar.png')
    else:
        print 'error, graph not fully connected, planarity test failed'
        ipdb.set_trace()
    ideal_max_face = []
    # ipdb.set_trace()
    for face in range(0, len(ideal_faces)):
        if len(ideal_faces[face]) > len(ideal_max_face):
            ideal_max_face = ideal_faces[face]
    ideal_pos = tutte_embedding(ideal_graph, ideal_max_face)

    nx.draw_networkx(ideal_graph, ideal_pos,node_color='r',node_size=25,alpha=0.2,edge_color='r')
    plt.xlim(-1.1,1.1)
    plt.ylim(-1.1, 1.1)
    plt.savefig(directory_name+'/'+ 'tutte_reconstruction_from_ideal.svg')
    plt.savefig(directory_name + '/' + 'tutte_reconstruction_from_ideal.png')
    plt.close()
    # ipdb.set_trace()

    vor_reconstructed = Voronoi(convert_dict_to_list(ideal_pos))
    voronoi_plot_2d(vor_reconstructed,show_vertices=False,show_points=False)
    plt.savefig(directory_name + '/' + 'voronoitutteideal.svg')
    plt.savefig(directory_name + '/' + 'voronoitutteideal.png')
    plt.close()


    #blind reconstruction here, if there are missed edges, then they will distort final reconstruction
    if (planarity.is_planar(All))==True and nx.is_connected(All)==True:
        faces = planar_embedding_draw(All,title='tutte')
    else:
        print 'error, graph not fully connected, planarity test failed'
        ipdb.set_trace()

    max_face = []

    for face in range(0,len(faces)):
        if len(faces[face]) > len(max_face):
            max_face = faces[face]
    tutte_pos = tutte_embedding(All, max_face)#need to update All
    ###########
    #All.add_edges_from(triangulation_fix(faces,tutte_pos))
    updated_tutte_pos=tutte_embedding(All,max_face)
    nx.draw_networkx(All, updated_tutte_pos,node_color='r',node_size=25)
    plt.xlim(-1.1,1.1)
    plt.ylim(-1.1, 1.1)
    plt.savefig(directory_name + '/' +'samseq_tutte_embedding.svg')
    plt.savefig(directory_name + '/' + 'samseq_tutte_embedding.png')
    plt.close()




    # ipdb.set_trace()

    rgb_stamp_catalog= (rgb_stamp_catalog)
    maxima = np.max(rgb_stamp_catalog)
    rgb_stamp_catalog = rgb_stamp_catalog/maxima
    vor_reconstructed = Voronoi(convert_dict_to_list(updated_tutte_pos))
    voronoi_plot_2d(vor_reconstructed,show_vertices=False,show_points=False)
    for r in range(len(vor_reconstructed.point_region)):
        region = vor_reconstructed.regions[vor_reconstructed.point_region[r]]
        if not -1 in region:
            polygon = [vor_reconstructed.vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=( rgb_stamp_catalog[r][0], rgb_stamp_catalog[r][1], rgb_stamp_catalog[r][2]))
    plt.xlim(-1.1,1.1)
    plt.ylim(-1.1, 1.1)
    plt.savefig(directory_name + '/' + 'voronoi_image_reconstructed_samseqtutte.svg')
    plt.savefig(directory_name + '/' + 'voronoi_image_reconstructed_samseqtutte.png')
    plt.close()

    spring_pos = nx.drawing.nx_agraph.pygraphviz_layout(All, args='-Goverlap=false')
    Border = nx.Graph()
    border_node = list()
    node_to_be_remove = list()
    border_node_bin = list()
    for i in range(0, len(cell_polony)):
        for k in cell_polony[i]:
            for j in range(i + 1, len(cell_polony)):
                for m in cell_polony[j]:
                    if k == m:
                        border_node.append(k)
                        node_to_be_remove.append(k)
                        border_node_bin.append(np.array([k, i, j]))
    #############################border marker##############################################3
    border_node = np.unique(border_node)
    pos_border = np.zeros((len(All.nodes), 2))
    for i in border_node:
        Border.add_node(int(i))
        pos_border[int(i)] = (spring_pos[int(i)])

    plt.figure()
    plt.axis('off')
    ####traingulation######
    All.add_edges_from(triangulation_fix(faces,spring_pos))
    nx.draw_networkx_nodes(Border, pos_border, node_size=100, node_color='gray', alpha=0.5)
    nx.draw_networkx_nodes(All, spring_pos, node_size=30, cmap=plt.cm.RdYlBu, node_color=polony_cell)

    nx.draw_networkx_edges(All, spring_pos, alpha=0.3)
    nx.draw_networkx_labels(All, spring_pos, font_family='Times', font_size=6, alpha=0.5)
    plt.savefig(directory_name + '/' + 'samseq_spring_embedding.svg')
    plt.savefig(directory_name + '/' + 'samseq_spring_embedding.png')

    vor_reconstructed_spring = Voronoi(convert_dict_to_list(spring_pos))
    voronoi_plot_2d(vor_reconstructed_spring,show_vertices=False,show_points=False)
    for r in range(len(vor_reconstructed_spring.point_region)):
        region = vor_reconstructed_spring.regions[vor_reconstructed_spring.point_region[r]]
        if not -1 in region:
            polygon = [vor_reconstructed_spring.vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=( rgb_stamp_catalog[r][0], rgb_stamp_catalog[r][1], rgb_stamp_catalog[r][2]))
    plt.savefig(directory_name + '/' + 'voronoi_image_reconstructed_samseqspring.svg')
    plt.savefig(directory_name + '/' + 'voronoi_image_reconstructed_samseqspring.png')
    plt.close()



    ############# convex hull polygon filling##################
    pos_cell_polony = list()
    for i in range(0, len(cell_polony)):
        subgroup = list()
        for j in cell_polony[i]:
            subgroup.append(spring_pos[int(j)])
        pos_cell_polony.append(subgroup)

    hull = list()
    for i in range(0, len(cell_polony)):
        if len(pos_cell_polony[i])>2:

            hull.append(ConvexHull(pos_cell_polony[i]))
    counter=0
    for i in range(0, len(cell_polony)):

        if len(pos_cell_polony[i])>2:

            cell = np.asarray(pos_cell_polony[i])

            plt.fill(cell[hull[counter].vertices, 0], cell[hull[counter].vertices, 1], lw=2, alpha=0.5, label='%s' % i)
            counter+=1
    centroid = np.zeros((len(cell_polony), 2))
    counter=0
    for i in range(0, len(cell_polony)):
        cell = np.asarray(pos_cell_polony[i])
        if len(pos_cell_polony[i])>2:
            centerx = sum(cell[hull[counter].vertices, 0]) / len(hull[counter].vertices)
            centery = sum(cell[hull[counter].vertices, 1]) / len(hull[counter].vertices)
            centroid[i] = [centerx, centery]
            counter+=1
    for i in range(0, len(cell_polony)):
        if len(pos_cell_polony[i])>2:
            plt.text(centroid[i, 0], centroid[i, 1], '%s' % i, fontweight='bold')

    plt.savefig(directory_name+'/'+ 'convexhull.svg')
    plt.close()
    return spring_pos, updated_tutte_pos, ideal_pos, All

def alignment_prcs(bc,hy,cell_point):
    score = np.zeros((len(hy), len(bc)))
    times = np.zeros((len(bc)))
    best = np.zeros((len(hy)))
    for i in range(0, len(hy)):
        for j in range(0, len(bc)):
            if bc[j] == hy[i][0:5]:
                best[i] = j
                times[j] += 1
        if j == len(bc) and best[i] == 0:  ###############avoid p2 fail to pair resulting in polony0 false positive
            best[i] = None

    np.savetxt(directory_name+'/'+ 'best', best, delimiter=",")
    best_2 = np.zeros((len(hy)))
    for i in range(0, len(hy)):
        for j in range(0, len(bc)):
            if bc[j] == hy[i][5:10]:
                best_2[i] = j
                times[j] += 1
        if j == len(bc) and best_2[i] == 0:  ###############avoid p2 fail to pair resulting in polony0 false positive
            best_2[i] = None
    best_2 = np.zeros((len(hy)))
    for i in range(0, len(hy)):
        for j in range(0, len(bc)):
            if bc[j] == hy[i][5:10]:
                best_2[i] = j

    np.savetxt(directory_name+'/'+ 'best_r', best_2, delimiter=",")
    ajc_polony = np.zeros((len(bc), len(bc)))
    ajc_polony.astype(int)
    for i in range(0, len(hy)):
        if best[i] != None and best_2[i] != None:
            r = int(best[i])
            c = int(best_2[i])
            ajc_polony[r, c] = 1
            ajc_polony[c, r] = 1

    np.savetxt(directory_name+'/'+ 'ajc_polony', ajc_polony, delimiter=",", fmt="%i")
    cell_hybrid = list()
    for i in range(0, len(cell_point)):
        subgroup = list()
        for j in cell_point[i]:  ######it's easy to get lost in index and value as index
            subgroup.append(hy[j])
        cell_hybrid.append(subgroup)

    cell_polony = list()
    for i in range(0, len(cell_hybrid)):
        subgroup = list()
        for j in cell_hybrid[i]:
            for k in j:
                subgroup.append(j[0])

        subgroup = np.unique(subgroup)
        subgroup = subgroup[~np.isnan(subgroup)]
        cell_polony.append(subgroup)
    polony_cell = np.zeros(len(bc))
    for i in range(0, len(cell_polony)):
        for j in cell_polony[i]:
            polony_cell[int(j)] = i

def circular_layout_sorted(graph,radius):
    #takes a circular graph, sorts them, and arranges them uniformly around a circle, returns the positions

    edges_to_sort = list(graph.edges())
    all_nodes = list(graph.nodes())
    maxpath = []
    for node in range(0, len(all_nodes)):
        for other_node in range(0, len(all_nodes)):
            try: path = max(nx.all_simple_paths(graph,all_nodes[node],all_nodes[other_node]))
            except: path = nx.all_simple_paths(graph,all_nodes[node],all_nodes[other_node])
            try:
                path_size = len(path)
                if path_size > len(maxpath):
                    maxpath = path
            except: pass

    pos = dict()
    delta_theta = 2*np.pi / len(maxpath)
    for arc in range(0,len(maxpath)):
        node_to_place = maxpath[arc]
        pos_x = np.cos(delta_theta*arc)
        pos_y = np.sin(delta_theta*arc)
        pos.update({node_to_place:(pos_x,pos_y)})
    return pos

def get_euclidean_edgelength(edge,positions):
    point_a = edge[0]
    # ipdb.set_trace()
    a_x = positions[point_a][0]
    a_y = positions[point_a][1]
    point_b = edge[1]
    b_x = positions[point_b][0]
    b_y = positions[point_b][1]
    edge_length = np.sqrt((a_x-b_x)**2+(a_y-b_y)**2)
    return edge_length

def positiondict_to_list(pos_dict):
    flatlist = []
    dictlistx = []
    dictlisty = []
    for key, value in pos_dict.iteritems():
        temp = [value[0],value[1]]
        flatlist.append(temp)
        dictlistx.append(value[0])
        dictlisty.append(value[1])
    flatlist = [item for sublist in flatlist for item in sublist]
    return flatlist, dictlistx, dictlisty



def separate_good_and_bad_pos(flat_pos_list,untethered_graph):
    pos_list = zip(flat_pos_list[::2], flat_pos_list[1::2])
    new_delaunay = Delaunay(pos_list)
    new_topology=[e for e in get_ideal_graph(new_delaunay).edges()]
    old_topology =[e for e in untethered_graph.edges()]
    good_edges = [edge for edge in new_topology if edge in old_topology]
    bad_new_edges = [edge for edge in new_topology if edge not in old_topology]
    missed_old_edges = [edge for edge in old_topology if edge not in new_topology]




    return bad_pos, good_pos, bad_pos_indices, good_pos_indices

# def get_delaunay_error_lite(flat_pos_list,untethered_graph):
#     # ipdb.set_trace()
#     pos_list = zip(flat_pos_list[::2], flat_pos_list[1::2])
#     # b = dict(zip(a[::2], a[1::2]))
#     pos = dict(zip(np.arange(len(pos_list)), pos_list))
#     new_delaunay = Delaunay(pos_list)
#     new_topology=[e for e in get_ideal_graph(new_delaunay).edges()]
#     old_topology =[e for e in untethered_graph.edges()]
#     good_edges = [edge for edge in new_topology if edge in old_topology]
#     bad_new_edges = [edge for edge in new_topology if edge not in old_topology]
#     missed_old_edges = [edge for edge in old_topology if edge not in new_topology]
#     missing_edge_penalty = 1
#     bad_edge_penalty = 1
#     good_edge_reward = len(good_edges)
#     for edge in missed_old_edges:
#         missing_edge_penalty += get_euclidean_edgelength(edge,pos)
#     # for edge in good_edges:
#     #     good_edge_reward += get_euclidean_edgelength(edge, pos)
#     for edge in bad_new_edges:
#         bad_edge_penalty += get_euclidean_edgelength(edge, pos)
#     cost = missing_edge_penalty - good_edge_reward + bad_edge_penalty
#     return cost

def callback_on_optimization_indicator(x, f, accepted):
    if f == 0:
        print(x, f, accepted)
        return True
    else:
        return False

def get_delaunay_error(flat_pos_list,untethered_graph):
    # ipdb.set_trace()
    pos_list = zip(flat_pos_list[::2], flat_pos_list[1::2])
    # b = dict(zip(a[::2], a[1::2]))
    pos = dict(zip(np.arange(len(pos_list)), pos_list))
    new_delaunay = Delaunay(pos_list)
    new_topology=[e for e in get_ideal_graph(new_delaunay).edges()]
    old_topology =[e for e in untethered_graph.edges()]
    good_edges = [edge for edge in new_topology if edge in old_topology]
    bad_new_edges = [edge for edge in new_topology if edge not in old_topology]
    missed_old_edges = [edge for edge in old_topology if edge not in new_topology]
    # ipdb.set_trace()
    average_distance = 0
    missing_edge_penalty = 1
    bad_edge_penalty = 1
    # good_edge_reward = 1
    for edge in new_topology:
        average_distance += get_euclidean_edgelength(edge,pos)
    average_distance = average_distance/len(new_topology)
    for edge in missed_old_edges:
        missing_edge_penalty += get_euclidean_edgelength(edge,pos)/len(missed_old_edges)
    # for edge in good_edges:
    #     good_edge_reward += get_euclidean_edgelength(edge, pos)
    for edge in bad_new_edges:
        bad_edge_penalty += get_euclidean_edgelength(edge, pos)/len(bad_new_edges)

    cost = (len(bad_new_edges)*bad_edge_penalty+len(missed_old_edges)*missing_edge_penalty)/(len(new_topology)*average_distance)
    return cost


def get_delaunay_comparison(flat_pos_list,untethered_graph,name=''):
    # ipdb.set_trace()
    pos_list = zip(flat_pos_list[::2], flat_pos_list[1::2])
    # b = dict(zip(a[::2], a[1::2]))
    pos = dict(zip(np.arange(len(pos_list)), pos_list))
    new_delaunay = Delaunay(pos_list)
    new_topology=[e for e in get_ideal_graph(new_delaunay).edges()]
    old_topology =[e for e in untethered_graph.edges()]
    good_edges = [edge for edge in new_topology if edge in old_topology]
    bad_new_edges = [edge for edge in new_topology if edge not in old_topology]
    missed_old_edges = [edge for edge in old_topology if edge not in new_topology]
    pos_x, pos_y = zip(*pos_list)[0],zip(*pos_list)[1]
    number_of_bad_edges = len(bad_new_edges)
    number_of_missed_edges = len(missed_old_edges)
    total_badedges = number_of_bad_edges + number_of_missed_edges
    # average_distance = 0
    # normalizing_factor = len(new_topology)
    # missing_edge_penalty = 1
    # bad_edge_penalty = len(bad_new_edges)
    # good_edge_reward = 1
    # for edge in new_topology:
    #     average_distance += get_euclidean_edgelength(edge,pos)
    # average_distance = average_distance / len(new_topology)
    # for edge in missed_old_edges:
    #     missing_edge_penalty += get_euclidean_edgelength(edge,pos)
    # for edge in good_edges:
    #     good_edge_reward += get_euclidean_edgelength(edge, pos)
    # for edge in bad_new_edges:
    #     bad_edge_penalty += get_euclidean_edgelength(edge, pos)


    plt.triplot(pos_x,pos_y,new_delaunay.simplices.copy(),linestyle=':')
    nx.draw_networkx(untethered_graph,pos=pos_list,linestyle=':',width=1.0,alpha=0.2,edge_color='r',node_color='r',node_size=30)
    plt.plot(pos_x,pos_y,'o')
    plt.rcParams['figure.figsize'] = (20, 20)
    for i in range(0, len(pos_list)):
        plt.text(pos_x[i], pos_y[i], '%s' % i, alpha=0.5)
    plt.xlim(-1.1,1.1)
    plt.ylim(-1.1, 1.1)
    plt.savefig(directory_name + '/' +'delaunay_badedges_'+str(total_badedges)+str(name)+'.svg')
    plt.savefig(directory_name + '/' + 'delaunay_badedges_'+str(total_badedges)+str(name)+'.png')
    plt.close()

    cost = (len(bad_new_edges) + len(missed_old_edges)) / len(new_topology)
    return good_edges,bad_new_edges,missed_old_edges,cost


def anneal_to_initial_delaunay(pos, untethered_graph, T=0.1,niter=1,name=''):
    pos_list= positiondict_to_list(pos)[0]
    # error = get_delaunay_error(flat_pos_list=tutte_pos_list,untethered_graph=All)
    minimizer_kwargs = {"tol":10000,"args": (untethered_graph,)}
    res = scipy.optimize.basinhopping(func=get_delaunay_error,
                                      x0=pos_list, T=T,
                                      minimizer_kwargs=minimizer_kwargs,
                                      niter=niter,
                                      disp = True,
                                      niter_success=50000,
                                      callback=callback_on_optimization_indicator)
    annealed_pos_list = zip(res.x[::2], res.x[1::2])
    annealed_pos = dict(zip(np.arange(len(annealed_pos_list)), annealed_pos_list))
    vor_reconstructed = Voronoi(convert_dict_to_list(annealed_pos))
    voronoi_plot_2d(vor_reconstructed,show_vertices=False,show_points=False)
    plt.xlim(-1.1,1.1)
    plt.ylim(-1.1, 1.1)
    plt.savefig(directory_name + '/' + 'annealed_voronoi'+str(name)+'.svg')
    plt.savefig(directory_name + '/' + 'annealed_voronoi'+str(name)+'.png')
    plt.close()


    good_edges, bad_edges, missed_edges,cost = get_delaunay_comparison(pos_list,untethered_graph,name=name+'before')
    good_edges_annealed, bad_edges_annealed, missed_edges_annealed,cost_annealed = get_delaunay_comparison(res.x, untethered_graph,name=name+'annealed')
    print 'input position discrepancies:'
    print 'missed edges: ' + str(missed_edges)
    print 'false edges: ' + str(bad_edges)
    print 'annealed position discrepancies:'
    print 'missed edges: ' + str(missed_edges_annealed)
    print 'false edges: ' + str(bad_edges_annealed)
    return annealed_pos