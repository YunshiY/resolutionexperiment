import simulator
import litesim
import numpy as np
import sys
import ipdb
import matplotlib.pyplot as plt
import random
import os
import litesim_lib
from select import select








#basic run with alignment
def basic_single_experiment_with_alignment():
    reload(litesim_lib)
    basic_circle, spring_reconstruction, tutte_reconstruction, spring_err, tutte_err = minimal_reconstruction_oop(target_nsites = 50 ,npolony = 25,directory_label = 'single')
    return basic_circle, spring_reconstruction, tutte_reconstruction, spring_err, tutte_err






def minimal_reconstruction_oop(target_nsites = 300 ,npolony = 50,filename = 'monalisacolor.png',master_directory = '', directory_label = '',randomseed=0,iterlabel=''):
    reload(litesim_lib)
    # directory_name = master_directory + '/' + directory_label
    if not master_directory:
        directory_name = litesim_lib.prefix(directory_label)
    else:
        directory_name = master_directory

    basic_circle = litesim_lib.Surface(target_nsites = target_nsites, directory_label='',master_directory=directory_name)
    basic_circle.circular_grid_ian()

    basic_circle.scale_image_axes(filename = 'monalisacolor.png')
    basic_circle.seed(nseed=npolony, full_output=False)
    basic_circle.crosslink_polonies(full_output=False)

    #ideal_graph = litesim.get_ideal_graph(delaunay)
    basic_circle.conduct_embeddings(full_output=False,image_only=False)

    spring_reconstruction = litesim_lib.Reconstruction(basic_circle.directory_name,
                    reconstructed_points=basic_circle.spring_points/ np.max(np.sqrt(basic_circle.spring_points ** 2)),
                                                       reconstructed_graph= basic_circle.untethered_graph,
                                                       corseed=basic_circle.corseed)
    spring_reconstruction.align(title='align_spring_points'+iterlabel, error_threshold=1, full_output=False)
    spring_reconstruction.get_delaunay_comparison(litesim_lib.positiondict_to_list(basic_circle.spring_pos)[0], basic_circle.untethered_graph)

    spring_err = litesim_lib.get_radial_profile(basic_circle.spring_points, spring_reconstruction.distances, title='rad_spring'+iterlabel)

    tutte_reconstruction = litesim_lib.Reconstruction(basic_circle.directory_name,
                                                       reconstructed_points=basic_circle.tutte_points,
                                                       reconstructed_graph=basic_circle.untethered_graph,
                                                       corseed=basic_circle.corseed)
    tutte_reconstruction.align(title='align_tutte_points' + iterlabel, error_threshold=1, full_output=False)
    tutte_reconstruction.get_delaunay_comparison(litesim_lib.positiondict_to_list(basic_circle.tutte_pos)[0],
                                                  basic_circle.untethered_graph)
    tutte_err = litesim_lib.get_radial_profile(basic_circle.tutte_points, tutte_reconstruction.distances, title='rad_tutte'+iterlabel)

    return basic_circle, spring_reconstruction, tutte_reconstruction, spring_err, tutte_err





reload(litesim_lib)
basic_circle, spring_reconstruction, tutte_reconstruction, spring_err, tutte_err = minimal_reconstruction_oop(target_nsites = 500 ,npolony = 25,directory_label = 'single')








def polony_number_variation_experiment():
    min = 10
    max = 50
    step = 10
    title = 'polony_variation'
    repeats_per_step = 3

    reload(litesim)
    master_directory = litesim_lib.prefix(title)
    number_of_steps = (max - min) / step

    master_errors_spring = np.zeros((number_of_steps * repeats_per_step))
    master_errors_tutte = np.zeros((number_of_steps * repeats_per_step))
    polony_counts = np.zeros((number_of_steps * repeats_per_step))

    for i in range(0, number_of_steps * repeats_per_step):
        npolony = min + i / repeats_per_step * step
        # substep = i%3
        print npolony
        polony_counts[i] = npolony
        # master_errors_spring[i], master_errors_tutte[i]
        basic_circle, spring_reconstruction, tutte_reconstruction, master_errors_spring[i], master_errors_tutte[
            i] = minimal_reconstruction_oop(target_nsites=500, npolony=npolony, randomseed=i % repeats_per_step,
                                            filename='monalisacolor.png', master_directory=master_directory,
                                            iterlabel=str(npolony))

        np.savetxt(master_directory + '/' + 'mastererrorsspring' + title + '.txt', zip(polony_counts, master_errors_spring))
        np.savetxt(master_directory + '/' + 'mastererrorstutte' + title + '.txt', zip(polony_counts, master_errors_tutte))

    timeout = 3
    print 'cd to ' + master_directory + '? (Y/n)'
    rlist, _, _ = select([sys.stdin], [], [], timeout)
    if rlist:
        cd_query = sys.stdin.readline()
        # ipdb.set_trace()
        print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
        if cd_query == 'y\n' or cd_query == 'yes\n' or cd_query == 'Y\n' or cd_query == 'YES\n' or cd_query == 'Yes\n' or cd_query == '\n':
            os.chdir(master_directory)
        else:
            print 'no cd'
    else:
        print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
        os.chdir(master_directory)

def site_number_variation_experiment():
    min = 100
    max = 1000
    step = 10
    title = 'polony_variation'
    repeats_per_step = 10

    reload(litesim)
    master_directory = litesim_lib.prefix(title)
    number_of_steps = (max - min) / step

    master_errors_spring = np.zeros((number_of_steps * repeats_per_step))
    master_errors_tutte = np.zeros((number_of_steps * repeats_per_step))
    site_counts = np.zeros((number_of_steps * repeats_per_step))

    for i in range(0, number_of_steps * repeats_per_step):
        nsites = min + i / repeats_per_step * step
        # substep = i%3
        print nsites
        site_counts[i] = nsites
        # master_errors_spring[i], master_errors_tutte[i]
        try: basic_circle, spring_reconstruction, tutte_reconstruction, master_errors_spring[i], master_errors_tutte[
            i] = minimal_reconstruction_oop(target_nsites=nsites, npolony=50, randomseed=i % repeats_per_step,
                                            filename='monalisacolor.png', master_directory=master_directory,
                                            iterlabel=str(nsites))
        except: pass
        np.savetxt(master_directory + '/' + 'mastererrorsspring' + title + '.txt', zip(site_counts, master_errors_spring))
        np.savetxt(master_directory + '/' + 'mastererrorstutte' + title + '.txt', zip(site_counts, master_errors_tutte))

    timeout = 3
    print 'cd to ' + master_directory + '? (Y/n)'
    rlist, _, _ = select([sys.stdin], [], [], timeout)
    if rlist:
        cd_query = sys.stdin.readline()
        # ipdb.set_trace()
        print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
        if cd_query == 'y\n' or cd_query == 'yes\n' or cd_query == 'Y\n' or cd_query == 'YES\n' or cd_query == 'Yes\n' or cd_query == '\n':
            os.chdir(master_directory)
        else:
            print 'no cd'
    else:
        print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
        os.chdir(master_directory)




def levenshtein_comparison_with_distortion():
    min = 250
    max = 260
    step = 10
    title = 'master_levenshtein_folder'
    repeats_per_step = 1000

    reload(litesim)
    master_directory = litesim_lib.prefix(title)
    number_of_steps = (max - min) / step

    master_errors_spring = np.zeros((number_of_steps * repeats_per_step))
    master_levenshtein_spring = np.zeros((number_of_steps * repeats_per_step,1))
    master_errors_tutte = np.zeros((number_of_steps * repeats_per_step))
    master_levenshtein_tutte = np.zeros((number_of_steps * repeats_per_step,1))
    polony_counts = np.zeros((number_of_steps * repeats_per_step))

    for i in range(0, number_of_steps * repeats_per_step):
        try:
            npolony = min + i / repeats_per_step * step
            nsites  = (min + i / repeats_per_step * step)*24
            # substep = i%3
            print npolony
            polony_counts[i] = npolony
            # master_errors_spring[i], master_errors_tutte[i]
            basic_circle, spring_reconstruction, tutte_reconstruction, master_errors_spring[i], master_errors_tutte[
                i] = minimal_reconstruction_oop(target_nsites=nsites, npolony=npolony, randomseed=i % repeats_per_step,
                                                filename='monalisacolor.png', master_directory=master_directory,
                                                iterlabel=str(npolony))

            master_levenshtein_spring[i] = spring_reconstruction.levenshtein_distance
            master_levenshtein_tutte[i] = tutte_reconstruction.levenshtein_distance
            # ipdb.set_trace()
            np.savetxt(master_directory + '/' + 'mastererrorsspring' + title + '.txt', zip(polony_counts, master_errors_spring))
            np.savetxt(master_directory + '/' + 'master_levenshtein_spring' + title + '.txt', zip(polony_counts, master_levenshtein_spring))
            np.savetxt(master_directory + '/' + 'mastererrorstutte' + title + '.txt', zip(polony_counts, master_errors_tutte))
            np.savetxt(master_directory + '/' + 'master_levenshtein_tutte' + title + '.txt', zip(polony_counts, master_levenshtein_tutte))
        except:pass

    timeout = 3
    print 'cd to ' + master_directory + '? (Y/n)'
    rlist, _, _ = select([sys.stdin], [], [], timeout)
    if rlist:
        cd_query = sys.stdin.readline()
        # ipdb.set_trace()
        print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
        if cd_query == 'y\n' or cd_query == 'yes\n' or cd_query == 'Y\n' or cd_query == 'YES\n' or cd_query == 'Yes\n' or cd_query == '\n':
            os.chdir(master_directory)
        else:
            print 'no cd'
    else:
        print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
        os.chdir(master_directory)




################################ <> <> <> <> <> <> <> <> <> <> <> <> ################################

#The basic starter template - use these lines to get a basic reconstruction dataset
#only construct image, embeddings,
#no alignment, no same-delaunay comparison

# reload(litesim_lib)
# #basic run
# target_nsites = 500
# npolony = 100
# basic_circle = litesim_lib.Surface(target_nsites = target_nsites, directory_label='oop_testing')
# basic_circle.circular_grid_ian()
# basic_circle.scale_image_axes(filename = 'monalisacolor.png')
# basic_circle.seed(nseed=npolony, full_output=False)
# basic_circle.crosslink_polonies(full_output=True)
# basic_circle.conduct_embeddings(full_output=True,image_only=False)

################################ <> <> <> <> <> <> <> <> <> <> <> <> ################################




# def vary_polony_no(xlen=100, ylen=100, min=10,max=50,step=10,title='polony_variation',repeats_per_step=3):
#     reload(litesim)
#     master_directory = litesim.prefix(title)
#     number_of_steps = (max-min)/step
#     master_errors_spring = np.zeros((number_of_steps*repeats_per_step))
#     master_errors_tutte = np.zeros((number_of_steps*repeats_per_step))
#     polony_counts = np.zeros((number_of_steps*repeats_per_step))
#     for i in range(0,number_of_steps*repeats_per_step):
#         npolony = min + i/repeats_per_step*step
#         # substep = i%3
#         print npolony
#         polony_counts[i] = npolony
#         try: master_errors_spring[i], master_errors_tutte[i] = minimal_reconstruction(xlen,ylen,ncell=1,npolony=npolony,randomseed=i%repeats_per_step,filename='monalisacolor.png',master_directory=master_directory,iterlabel=str(npolony))
#         except: pass
#         np.savetxt(master_directory + '/' + 'mastererrorsspring' + title + '.txt', zip(polony_counts, master_errors_spring))
#         np.savetxt(master_directory + '/' + 'mastererrorstutte' + title + '.txt', zip(polony_counts, master_errors_tutte))
#
#     timeout = 3
#     print 'cd to ' + master_directory + '? (Y/n)'
#     rlist, _, _ = select([sys.stdin], [], [], timeout)
#     if rlist:
#         cd_query = sys.stdin.readline()
#         # ipdb.set_trace()
#         print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
#         if cd_query == 'y\n' or cd_query =='yes\n' or cd_query =='Y\n' or cd_query =='YES\n'  or cd_query =='Yes\n' or cd_query =='\n':
#             os.chdir(master_directory)
#         else:print 'no cd'
#     else:
#         print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
#         os.chdir(master_directory)

# def minimal_reconstruction(xlen,ylen,ncell,npolony,filename = 'monalisacolor.png',master_directory = '', directory_label = '',randomseed=0,iterlabel=''):
#     reload(litesim)
#     # directory_name = master_directory + '/' + directory_label
#     if not master_directory:
#         directory_name = litesim.prefix(directory_label)
#     else:
#         directory_name = master_directory
#     points = litesim.circular_grid_yunshi(xlen, ylen, randomseed=randomseed)
#     scaled_imarray, lookup_x_axis, lookup_y_axis = litesim.scale_image_axes(points, filename = filename)
#     corseed, p1, bc, seed = litesim.seed(npolony, points,full_output=False)
#     p2,p2_sec,p2_prim,cheating_sheet,delaunay,rgb_stamp_catalog=litesim.rca(p1,points,corseed,scaled_imarray, lookup_x_axis, lookup_y_axis,full_output=False)
#     ideal_graph = litesim.get_ideal_graph(delaunay)
#     spring_pos, tuttesamseq_pos, ideal_pos, untethered_graph = litesim.cheating_alignment(cheating_sheet, bc,
#                                                                                             group_cell=0, ideal_graph=ideal_graph,
#                                                                                             rgb_stamp_catalog=rgb_stamp_catalog,
#                                                                                           full_output=False)
#     spring_points = np.array(litesim.convert_dict_to_list(spring_pos))
#     spring_distances = litesim.align(corseed, spring_points / np.max(np.sqrt(spring_points ** 2)),
#                                        reconstructed_graph=untethered_graph, title='align_spring_points'+iterlabel,
#                                        error_threshold=1, full_output=False)
#     spring_err =  litesim.get_radial_profile(spring_points, spring_distances, title='rad_spring'+iterlabel)
#
#     tuttesamseq_points = np.array(litesim.convert_dict_to_list(tuttesamseq_pos))
#     tuttesamseq_distances = litesim.align(corseed, tuttesamseq_points, reconstructed_graph=untethered_graph,
#                                             title='align_tuttesamseq_points'+iterlabel, error_threshold=1.,
#                                             edge_number=len(tuttesamseq_points) * 3, full_output=False)
#     tutte_err = litesim.get_radial_profile(tuttesamseq_points, tuttesamseq_distances, title='rad_tutte'+iterlabel)
#     return spring_err, tutte_err
#
#     # timeout = 3
#     # print 'cd to ' + directory_name + '? (Y/n)'
#     # rlist, _, _ = select([sys.stdin], [], [], timeout)
#     # if rlist:
#     #     cd_query = sys.stdin.readline()
#     #     # ipdb.set_trace()
#     #     print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
#     #     if cd_query == 'y\n' or cd_query =='yes\n' or cd_query =='Y\n' or cd_query =='YES\n'  or cd_query =='Yes\n' or cd_query =='\n':
#     #         os.chdir(directory_name)
#     #     else:print 'no cd'
#     # else:
#     #     print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
#     #     os.chdir(directory_name)

def image_only_reconstruct(xlen,ylen,ncell,npolony,filename = 'monalisacolor.png',master_directory = '', directory_label = '',randomseed=0,iterlabel=''):
    reload(litesim)
    # directory_name = master_directory + '/' + directory_label
    if not master_directory:
        directory_name = litesim.prefix(directory_label)
    else:
        directory_name = master_directory
    points = litesim.circular_grid_yunshi(xlen, ylen, randomseed=randomseed)
    scaled_imarray, lookup_x_axis, lookup_y_axis = litesim.scale_image_axes(points, filename = filename)
    corseed, p1, bc, seed = litesim.seed(npolony, points,full_output=False)
    p2,p2_sec,p2_prim,cheating_sheet,delaunay,rgb_stamp_catalog=litesim.rca(p1,points,corseed,scaled_imarray, lookup_x_axis, lookup_y_axis,full_output=False)
    ideal_graph = litesim.get_ideal_graph(delaunay)
    spring_pos, tuttesamseq_pos, ideal_pos, untethered_graph = litesim.cheating_alignment(cheating_sheet, bc,
                                                                                            group_cell=0, ideal_graph=ideal_graph,
                                                                                            rgb_stamp_catalog=rgb_stamp_catalog,
                                                                                          full_output=False, image_only = True)
    spring_points = np.array(litesim.convert_dict_to_list(spring_pos))
    spring_distances = litesim.align(corseed, spring_points / np.max(np.sqrt(spring_points ** 2)),
                                       reconstructed_graph=untethered_graph, title='align_spring_points'+iterlabel,
                                       error_threshold=1, full_output=False, image_only=True)
    spring_err =  litesim.get_radial_profile(spring_points, spring_distances, title='rad_spring'+iterlabel)

    tuttesamseq_points = np.array(litesim.convert_dict_to_list(tuttesamseq_pos))
    tuttesamseq_distances = litesim.align(corseed, tuttesamseq_points, reconstructed_graph=untethered_graph,
                                            title='align_tuttesamseq_points'+iterlabel, error_threshold=1.,
                                            edge_number=len(tuttesamseq_points) * 3, full_output=False,image_only=True)
    tutte_err = litesim.get_radial_profile(tuttesamseq_points, tuttesamseq_distances, title='rad_tutte'+iterlabel)
    return spring_err, tutte_err

def full_experiment(xlen,ylen,ncell,npolony,filename = 'monalisacolor.png',directory_label = '',niter=1,randomseed=4):
    reload(litesim)
    directory_name = litesim.prefix(directory_label)
    points=litesim.circular_grid_yunshi(xlen,ylen,randomseed=randomseed)
    scaled_imarray, lookup_x_axis, lookup_y_axis = litesim.scale_image_axes(points, filename = filename)

    corseed,p1,bc,seed=litesim.seed(npolony,points)
    p2,p2_sec,p2_prim,cheating_sheet,delaunay,rgb_stamp_catalog=simulator.rca(p1,points,corseed,scaled_imarray, lookup_x_axis, lookup_y_axis)
    ideal_graph = litesim.get_ideal_graph(delaunay)
    hybrid=litesim.perfect_hybrid(p2,p2_sec,bc)
    cell_points=litesim.cell(points,ncell,corseed,hybrid,xlen,ylen)
    spring_pos, tuttesamseq_pos, ideal_pos, untethered_graph = litesim.cheating_alignment(cheating_sheet,bc,cell_points,ideal_graph,rgb_stamp_catalog,points,npolony)
    # flat_tutte_pos, dictlistx, dictlisty = simulator.positiondict_to_list(tuttesamseq_pos)
    # good_edges, bad_new_edges, missed_old_edges, cost = simulator.get_delaunay_comparison(simulator.positiondict_to_list(spring_pos)[0], untethered_graph)
    # print cost
    annealed_pos_spring = litesim.anneal_to_initial_delaunay(pos=spring_pos, untethered_graph = untethered_graph,niter=niter,name='spring')
    annealed_pos_tutte = litesim.anneal_to_initial_delaunay(pos=tuttesamseq_pos, untethered_graph=untethered_graph, niter=niter,name='tutte')
    annealed_pos_tutte_ideal = litesim.anneal_to_initial_delaunay(pos=ideal_pos, untethered_graph=ideal_graph,
                                                              niter=niter, name='tutteideal')

    ### get alignments with original positions ###
    edge_number = len(cheating_sheet)
    ideal_points = np.array(litesim.convert_dict_to_list(ideal_pos))
    ideal_distances = litesim.align(corseed,ideal_points,title='align_ideal_points',error_threshold = 1.,edge_number = len(ideal_points)*3)
    litesim.get_radial_profile(ideal_points,ideal_distances,title='radial_error_plot_ideal')

    spring_points = np.array(litesim.convert_dict_to_list(spring_pos))
    spring_distances = litesim.align(corseed,spring_points/np.max(np.sqrt(spring_points**2)),reconstructed_graph = untethered_graph,title='align_spring_points',error_threshold = 1)
    litesim.get_radial_profile(spring_points, spring_distances, title='radial_error_plot_spring')

    tuttesamseq_points = np.array(litesim.convert_dict_to_list(tuttesamseq_pos))
    tuttesamseq_distances = litesim.align(corseed,tuttesamseq_points,reconstructed_graph = untethered_graph, title='align_tuttesamseq_points',error_threshold = 1.,edge_number = len(ideal_points)*3)
    litesim.get_radial_profile(tuttesamseq_points,tuttesamseq_distances,title='radial_error_plot_tuttesamseq')

    annealedsamseq_points_spring = np.array(litesim.convert_dict_to_list(annealed_pos_spring))
    annealedsamseq_distances_spring = litesim.align(corseed,annealedsamseq_points_spring/np.max(np.sqrt(annealedsamseq_points_spring**2)),reconstructed_graph = untethered_graph, title='align_annealedsamseq_points_spring',error_threshold = 1.,edge_number = len(ideal_points)*3)
    litesim.get_radial_profile(annealedsamseq_points_spring,annealedsamseq_distances_spring,title='radial_error_plot_annealedsamseq')
    # ipdb.set_trace()
    annealedsamseq_points_tutte = np.array(litesim.convert_dict_to_list(annealed_pos_tutte))
    annealedsamseq_distances_tutte = litesim.align(corseed,annealedsamseq_points_tutte/np.max(np.sqrt(annealedsamseq_points_tutte**2)),reconstructed_graph = untethered_graph, title='align_annealedsamseq_points_tutte',error_threshold = 1.,edge_number = len(ideal_points)*3)
    litesim.get_radial_profile(annealedsamseq_points_tutte,annealedsamseq_distances_tutte,title='radial_error_plot_annealedsamseq')

    annealedsamseq_points_tutte_ideal = np.array(litesim.convert_dict_to_list(annealed_pos_tutte_ideal))
    annealedsamseq_distances_tutte_ideal = litesim.align(corseed, annealedsamseq_points_tutte_ideal / np.max(
        np.sqrt(annealedsamseq_points_tutte_ideal ** 2)), reconstructed_graph=ideal_graph,
                                                     title='align_annealedsamseq_points_tutte_ideal', error_threshold=1.,
                                                     edge_number=len(ideal_points) * 3)
    litesim.get_radial_profile(annealedsamseq_points_tutte_ideal, annealedsamseq_distances_tutte_ideal,
                                 title='radial_error_plot_annealedsamseq_ideal')


    #change to the file directory at the end
    timeout = 10
    print 'cd to ' + directory_name + '? (Y/n)'
    rlist, _, _ = select([sys.stdin], [], [], timeout)
    if rlist:
        cd_query = sys.stdin.readline()
        # ipdb.set_trace()
        print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
        if cd_query == 'y\n' or cd_query =='yes\n' or cd_query =='Y\n' or cd_query =='YES\n'  or cd_query =='Yes\n' or cd_query =='\n':
            os.chdir(directory_name)
        else:print 'no cd'
    else:
        print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
        os.chdir(directory_name)
    # ipdb.set_trace()

