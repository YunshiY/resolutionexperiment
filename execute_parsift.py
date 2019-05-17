import simulator
import litesim
import numpy as np
import sys
import ipdb
import matplotlib.pyplot as plt
import random
import os
import parsift_lib
from select import select
import time


class Experiment:
    def __init__(self, directory_label = ''):
        self.directory_label = directory_label

    def reconstruct(self,target_nsites=300, npolony=50, filename='monalisacolor.png', master_directory='',
                                   directory_label='', randomseed=0, iterlabel='',full_output=False, do_peripheral_pca = True,do_total_pca = True,do_spring = True,do_tutte = True):
        reload(parsift_lib)
        # directory_name = master_directory + '/' + directory_label
        if not master_directory:
            directory_name = parsift_lib.prefix(directory_label)
        else:
            directory_name = master_directory

        ########## SURFACE SIMULATION STEPS ##########
        t0 = time.time()
        self.basic_circle = parsift_lib.Surface(target_nsites=target_nsites, directory_label='',
                                           master_directory=directory_name)
        self.basic_circle.circular_grid_ian()
        self.basic_circle.scale_image_axes(filename='monalisacolor.png')
        self.basic_circle.seed(nseed=npolony, full_output=full_output)
        self.basic_circle.crosslink_polonies(full_output=full_output)
        surface_prep_t = time.time() - t0




        if do_peripheral_pca == True:
            ########## PERIPHERAL PCA RECONSTRUCTION ##########
            self.peripheralpca_reconstruction = parsift_lib.Reconstruction(self.basic_circle.directory_name,
                                                               corseed=self.basic_circle.corseed,
                                                               ideal_graph=self.basic_circle.ideal_graph,
                                                               untethered_graph=self.basic_circle.untethered_graph,
                                                               rgb_stamp_catalog=self.basic_circle.rgb_stamp_catalog)
            peripheralpca_time0 = time.time()
            self.peripheralpca_reconstruction.conduct_pca_peripheral(full_output=full_output, image_only=False)
            self.peripheralpca_time = time.time() - peripheralpca_time0
            self.peripheralpca_reconstruction.align(title='align_peripheralpca_points' + iterlabel, error_threshold=1, full_output=full_output)
            self.peripheralpca_reconstruction.get_delaunay_comparison(parsift_lib.positiondict_to_list(self.peripheralpca_reconstruction.reconstructed_pos)[0], self.basic_circle.untethered_graph)
            self.peripheralpca_err = parsift_lib.get_radial_profile(self.peripheralpca_reconstruction.reconstructed_points, self.peripheralpca_reconstruction.distances, title='rad_peripheralpca' + iterlabel)



        if do_total_pca == True:
            ########## TOTAL PCA RECONSTRUCTION ##########
            self.totalpca_reconstruction = parsift_lib.Reconstruction(self.basic_circle.directory_name,
                                                               corseed=self.basic_circle.corseed,
                                                               ideal_graph=self.basic_circle.ideal_graph,
                                                               untethered_graph=self.basic_circle.untethered_graph,
                                                               rgb_stamp_catalog=self.basic_circle.rgb_stamp_catalog)
            totalpca_time0 = time.time()
            self.totalpca_reconstruction.conduct_shortest_path_matrix(full_output=full_output, image_only=False)
            self.totalpca_time = time.time() - totalpca_time0
            self.totalpca_reconstruction.align(title='align_totalpca_points' + iterlabel, error_threshold=1, full_output=full_output)
            self.totalpca_reconstruction.get_delaunay_comparison(parsift_lib.positiondict_to_list(self.totalpca_reconstruction.reconstructed_pos)[0], self.basic_circle.untethered_graph)
            self.totalpca_err = parsift_lib.get_radial_profile(self.totalpca_reconstruction.reconstructed_points,
                                                         self.totalpca_reconstruction.distances, title='rad_totalpca' + iterlabel)

        if do_spring == True:
            ########## SPRING RECONSTRUCTION ##########
            self.spring_reconstruction = parsift_lib.Reconstruction(self.basic_circle.directory_name,
                                                               corseed=self.basic_circle.corseed,
                                                               ideal_graph=self.basic_circle.ideal_graph,
                                                               untethered_graph=self.basic_circle.untethered_graph,
                                                               rgb_stamp_catalog=self.basic_circle.rgb_stamp_catalog)
            springt0 = time.time()
            self.spring_reconstruction.conduct_spring_embedding(full_output=full_output, image_only=False)
            # ipdb.set_trace()
            self.springtime = time.time() - springt0

            self.spring_reconstruction.align(title='align_spring_points' + iterlabel, error_threshold=1, full_output=full_output)
            self.spring_reconstruction.get_delaunay_comparison(
                parsift_lib.positiondict_to_list(self.spring_reconstruction.reconstructed_pos)[0], self.basic_circle.untethered_graph)

            self.spring_err = parsift_lib.get_radial_profile(self.spring_reconstruction.reconstructed_points,
                                                        self.spring_reconstruction.distances, title='rad_spring' + iterlabel)

        if do_tutte == True:
            ########## TUTTE RECONSTRUCTION ##########
            tuttet0 = time.time()
            self.tutte_reconstruction = parsift_lib.Reconstruction(self.basic_circle.directory_name,
                                                               corseed=self.basic_circle.corseed,
                                                               ideal_graph=self.basic_circle.ideal_graph,
                                                               untethered_graph=self.basic_circle.untethered_graph,
                                                               rgb_stamp_catalog=self.basic_circle.rgb_stamp_catalog)
            self.tutte_reconstruction.conduct_tutte_embedding(full_output=full_output, image_only=False)
            #tutte_reconstruction.face_enumeration_runtime
            self.tuttetime = time.time() - tuttet0
            self.tutte_reconstruction.align(title='align_tutte_points' + iterlabel, error_threshold=1, full_output=full_output)
            self.tutte_reconstruction.get_delaunay_comparison(
                 parsift_lib.positiondict_to_list(self.tutte_reconstruction.reconstructed_pos)[0],
                 self.basic_circle.untethered_graph)
            self.tutte_err = parsift_lib.get_radial_profile(self.tutte_reconstruction.reconstructed_points,
                                                        self.tutte_reconstruction.distances, title='rad_tutte' + iterlabel)
            self.tutte_reconstruction.conduct_tutte_embedding_from_ideal(full_output=full_output, image_only=False)









#basic run with alignment
def basic_single_experiment_with_alignment(target_nsites=1000,npolony=100):
    reload(parsift_lib)
    start_t = time.time()
    basic_run = Experiment(directory_label='single')
    basic_run.reconstruct(target_nsites=target_nsites, npolony=npolony, full_output=True, do_peripheral_pca=False, do_total_pca=False,
                          do_spring=False, do_tutte=True)
    run_time = time.time() - start_t
    print run_time
#
# basic_single_experiment_with_alignment(target_nsites=200000,npolony=10000)
# print '10000'
#
# basic_single_experiment_with_alignment(target_nsites=120000,npolony=12000)
# print '12000'
#
# basic_single_experiment_with_alignment(target_nsites=140000,npolony=14000)





def polony_number_variation_experiment():
    min = 10
    max = 2000
    step = 10
    title = 'polony_variation'
    repeats_per_step = 1

    reload(parsift_lib)
    master_directory = parsift_lib.prefix(title)
    number_of_steps = (max - min) / step
    polony_counts = np.zeros((number_of_steps * repeats_per_step))


    master_errors_peripheralpca = np.zeros((number_of_steps * repeats_per_step))
    master_errors_totalpca = np.zeros((number_of_steps * repeats_per_step))
    master_errors_spring = np.zeros((number_of_steps * repeats_per_step))
    master_errors_tutte = np.zeros((number_of_steps * repeats_per_step))

    master_levenshtein_peripheralpca = np.zeros((number_of_steps * repeats_per_step, 1))
    master_levenshtein_totalpca = np.zeros((number_of_steps * repeats_per_step, 1))
    master_levenshtein_spring = np.zeros((number_of_steps * repeats_per_step, 1))
    master_levenshtein_tutte = np.zeros((number_of_steps * repeats_per_step, 1))


    peripheralpca_run_times = np.zeros((number_of_steps * repeats_per_step))
    totalpca_run_times = np.zeros((number_of_steps * repeats_per_step))
    spring_run_times = np.zeros((number_of_steps * repeats_per_step))
    tutte_run_times = np.zeros((number_of_steps * repeats_per_step))
    face_enumeration_run_times = np.zeros((number_of_steps * repeats_per_step))

    for i in range(0, number_of_steps * repeats_per_step):
        try:
            npolony = min + i / repeats_per_step * step
            nsites = (min + i / repeats_per_step * step) * 24
            # substep = i%3
            print npolony
            polony_counts[i] = npolony
            # master_errors_spring[i], master_errors_tutte[i]
            # basic_circle, spring_reconstruction, tutte_reconstruction, master_errors_spring[i], master_errors_tutte[
            #     i], spring_run_times[i], tutte_run_times[i],face_enumeration_run_times[i] = minimal_reconstruction_oop(target_nsites=nsites, npolony=npolony, randomseed=i % repeats_per_step,
            #                                     filename='monalisacolor.png', master_directory=master_directory,
            #                                     iterlabel=str(npolony))

            basic_run = Experiment(directory_label='single')
            basic_run.reconstruct(target_nsites=nsites, npolony=npolony,randomseed=i % repeats_per_step,filename='monalisacolor.png', master_directory=master_directory,
                                                iterlabel=str(npolony), full_output=False, do_peripheral_pca=False, do_total_pca=False,
                          do_spring=False, do_tutte=True)

            try:
                master_errors_peripheralpca[i] = basic_run.peripheralpca_err
            except: pass
            try:master_errors_totalpca[i] = basic_run.totalpca_err
            except: pass
            try:master_errors_spring[i] = basic_run.spring_err
            except: pass
            try:master_errors_tutte[i] = basic_run.tutte_err
            except: pass

            try:peripheralpca_run_times[i] = basic_run.peripheralpca_time
            except: pass
            try:totalpca_run_times[i] = basic_run.totalpca_time
            except: pass
            try:spring_run_times[i] = basic_run.springtime
            except: pass
            try:tutte_run_times[i] = basic_run.tuttetime
            except: pass
            try:face_enumeration_run_times[i] = basic_run.tutte_reconstruction.face_enumeration_runtime
            except: pass

            try:master_levenshtein_peripheralpca[i] = basic_run.peripheralpca_reconstruction.levenshtein_distance
            except: pass
            try:master_levenshtein_totalpca[i] = basic_run.totalpca_reconstruction.levenshtein_distance
            except: pass
            try:master_levenshtein_spring[i] = basic_run.spring_reconstruction.levenshtein_distance
            except: pass
            try:master_levenshtein_tutte[i] = basic_run.tutte_reconstruction.levenshtein_distance
            except: pass
            # ipdb.set_trace()


            try:np.savetxt(master_directory + '/' + 'master_levenshtein_peripheralpca' + title + '.txt', zip(polony_counts, master_levenshtein_peripheralpca))
            except: pass
            try:np.savetxt(master_directory + '/' + 'master_levenshtein_totalpca' + title + '.txt', zip(polony_counts, master_levenshtein_totalpca))
            except: pass
            try:np.savetxt(master_directory + '/' + 'master_levenshtein_spring' + title + '.txt', zip(polony_counts, master_levenshtein_spring))
            except: pass
            try:np.savetxt(master_directory + '/' + 'master_levenshtein_tutte' + title + '.txt', zip(polony_counts, master_levenshtein_tutte))
            except: pass

            try:np.savetxt(master_directory + '/' + '0mastererrorsperipheralpca' + title + '.txt', zip(polony_counts, master_errors_peripheralpca))
            except: pass
            try:np.savetxt(master_directory + '/' + '0mastererrorstotalpca' + title + '.txt', zip(polony_counts, master_errors_totalpca))
            except: pass
            try:np.savetxt(master_directory + '/' + '0mastererrorsspring' + title + '.txt', zip(polony_counts, master_errors_spring))
            except: pass
            try:np.savetxt(master_directory + '/' + '0mastererrorstutte' + title + '.txt', zip(polony_counts, master_errors_tutte))
            except: pass

            try:np.savetxt(master_directory + '/' + '1peripheralpca_runtimes_' + title + '.txt', zip(polony_counts, peripheralpca_run_times))
            except: pass
            try:np.savetxt(master_directory + '/' + '1totalpca_runtimes_' + title + '.txt', zip(polony_counts, totalpca_run_times))
            except: pass
            try:np.savetxt(master_directory + '/' + '1spring_runtimes_' + title + '.txt', zip(polony_counts, spring_run_times))
            except: pass
            try:np.savetxt(master_directory + '/' + '1tutte_runtimes_' + title + '.txt', zip(polony_counts, tutte_run_times))
            except: pass

            try:np.savetxt(master_directory + '/' + '1face_enumeration_runtimes_' + title + '.txt', zip(polony_counts, face_enumeration_run_times))
            except: pass
        except:
            print 'ERROR'
            # ipdb.set_trace()
            pass

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
    max = 110
    step = 10
    title = 'site_variation'
    repeats_per_step = 1

    reload(parsift_lib)
    master_directory = parsift_lib.prefix(title)
    number_of_steps = (max - min) / step
    site_counts = np.zeros((number_of_steps * repeats_per_step))

    master_errors_peripheralpca = np.zeros((number_of_steps * repeats_per_step))
    master_errors_totalpca = np.zeros((number_of_steps * repeats_per_step))
    master_errors_spring = np.zeros((number_of_steps * repeats_per_step))
    master_errors_tutte = np.zeros((number_of_steps * repeats_per_step))
    polony_counts = np.zeros((number_of_steps * repeats_per_step))

    peripheralpca_run_times = np.zeros((number_of_steps * repeats_per_step))
    totalpca_run_times = np.zeros((number_of_steps * repeats_per_step))
    spring_run_times = np.zeros((number_of_steps * repeats_per_step))
    tutte_run_times = np.zeros((number_of_steps * repeats_per_step))
    face_enumeration_run_times = np.zeros((number_of_steps * repeats_per_step))

    for i in range(0, number_of_steps * repeats_per_step):
        try:
            nsites = min + i / repeats_per_step * step
            # substep = i%3
            print nsites
            site_counts[i] = nsites
            # master_errors_spring[i], master_errors_tutte[i]
            # try:
                # basic_circle, spring_reconstruction, tutte_reconstruction, master_errors_spring[i], master_errors_tutte[
            #     i] = minimal_reconstruction_oop(target_nsites=nsites, npolony=50, randomseed=i % repeats_per_step,
            #                                     filename='monalisacolor.png', master_directory=master_directory,
            #                                     iterlabel=str(nsites))
            basic_run = Experiment(directory_label='single')
            basic_run.reconstruct(target_nsites=nsites, npolony=50,randomseed=i % repeats_per_step,filename='monalisacolor.png', master_directory=master_directory,
                                                iterlabel=str(nsites), full_output=False, do_peripheral_pca=True, do_total_pca=True,
                      do_spring=True, do_tutte=True)
            master_errors_peripheralpca[i] = basic_run.peripheralpca_err
            master_errors_totalpca[i] = basic_run.totalpca_err
            master_errors_spring[i] = basic_run.spring_err
            master_errors_tutte[i] = basic_run.tutte_err


            peripheralpca_run_times[i] = basic_run.peripheralpca_time
            totalpca_run_times[i] = basic_run.totalpca_time
            spring_run_times[i] = basic_run.springtime
            tutte_run_times[i] = basic_run.tuttetime
            face_enumeration_run_times[i] = basic_run.tutte_reconstruction.face_enumeration_runtime

            np.savetxt(master_directory + '/' + 'mastererrorsperipheralpca' + title + '.txt',
                       zip(polony_counts, master_errors_peripheralpca))
            np.savetxt(master_directory + '/' + 'mastererrorstotalpca' + title + '.txt',
                       zip(polony_counts, master_errors_totalpca))
            np.savetxt(master_directory + '/' + 'mastererrorsspring' + title + '.txt',
                       zip(polony_counts, master_errors_spring))
            np.savetxt(master_directory + '/' + 'mastererrorstutte' + title + '.txt',
                       zip(polony_counts, master_errors_tutte))

            np.savetxt(master_directory + '/' + 'peripheralpca_runtimes_' + title + '.txt',
                       zip(polony_counts, peripheralpca_run_times))
            np.savetxt(master_directory + '/' + 'totalpca_runtimes_' + title + '.txt',
                       zip(polony_counts, totalpca_run_times))
            np.savetxt(master_directory + '/' + 'spring_runtimes_' + title + '.txt',
                       zip(polony_counts, spring_run_times))
            np.savetxt(master_directory + '/' + 'tutte_runtimes_' + title + '.txt', zip(polony_counts, tutte_run_times))

            np.savetxt(master_directory + '/' + 'face_enumeration_runtimes_' + title + '.txt',
                       zip(polony_counts, face_enumeration_run_times))
        except:
            print 'error reconstructing with the given parameters'

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
    repeats_per_step = 1

    reload(parsift_lib)
    master_directory = parsift_lib.prefix(title)
    number_of_steps = (max - min) / step
    polony_counts = np.zeros((number_of_steps * repeats_per_step))

    master_errors_spring = np.zeros((number_of_steps * repeats_per_step))
    master_levenshtein_spring = np.zeros((number_of_steps * repeats_per_step,1))
    master_errors_tutte = np.zeros((number_of_steps * repeats_per_step))
    master_levenshtein_tutte = np.zeros((number_of_steps * repeats_per_step,1))


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









# def minimal_reconstruction_oop(target_nsites = 300 ,npolony = 50,filename = 'monalisacolor.png',master_directory = '', directory_label = '',randomseed=0,iterlabel=''):
#     reload(parsift_lib)
#     # directory_name = master_directory + '/' + directory_label
#     if not master_directory:
#         directory_name = parsift_lib.prefix(directory_label)
#     else:
#         directory_name = master_directory
#     t0 = time.time()
#     basic_circle = parsift_lib.Surface(target_nsites = target_nsites, directory_label='',master_directory=directory_name)
#     basic_circle.circular_grid_ian()
#     basic_circle.scale_image_axes(filename = 'monalisacolor.png')
#     basic_circle.seed(nseed=npolony, full_output=False)
#     basic_circle.crosslink_polonies(full_output=False)
#     surface_prep_t = time.time() - t0
#
#
#     #ideal_graph = litesim.get_ideal_graph(delaunay)
#     # basic_circle.conduct_embeddings(full_output=False,image_only=False)
#
#     spring_reconstruction = parsift_lib.Reconstruction(basic_circle.directory_name,
#                                                        corseed=basic_circle.corseed,
#                                                        ideal_graph = basic_circle.ideal_graph,
#                                                        untethered_graph = basic_circle.untethered_graph,
#                                                        rgb_stamp_catalog = basic_circle.rgb_stamp_catalog)
#     springt0 = time.time()
#     spring_reconstruction.conduct_spring_embedding(full_output=False,image_only=False)
#     springtime = time.time() - springt0
#
#     spring_reconstruction.align(title='align_spring_points'+iterlabel, error_threshold=1, full_output=True)
#     spring_reconstruction.get_delaunay_comparison(parsift_lib.positiondict_to_list(spring_reconstruction.reconstructed_pos)[0], basic_circle.untethered_graph)
#
#     spring_err = parsift_lib.get_radial_profile(spring_reconstruction.reconstructed_points, spring_reconstruction.distances, title='rad_spring'+iterlabel)
#     tuttet0 = time.time()
#     tutte_reconstruction = parsift_lib.Reconstruction(basic_circle.directory_name,
#                                                        corseed=basic_circle.corseed,
#                                                        ideal_graph = basic_circle.ideal_graph,
#                                                        untethered_graph = basic_circle.untethered_graph,
#                                                        rgb_stamp_catalog = basic_circle.rgb_stamp_catalog)
#
#     tutte_reconstruction.conduct_tutte_embedding(full_output=False, image_only=False)
#     # tutte_reconstruction.face_enumeration_runtime
#     tuttetime = time.time() - tuttet0
#
#     tutte_reconstruction.align(title='align_tutte_points' + iterlabel, error_threshold=1, full_output=False)
#     tutte_reconstruction.get_delaunay_comparison(parsift_lib.positiondict_to_list(tutte_reconstruction.reconstructed_pos)[0],
#                                                   basic_circle.untethered_graph)
#     tutte_err = parsift_lib.get_radial_profile(tutte_reconstruction.reconstructed_points, tutte_reconstruction.distances, title='rad_tutte'+iterlabel)
#
#     tutte_reconstruction.conduct_tutte_embedding_from_ideal(full_output=False,image_only=False)
#
#     return basic_circle, spring_reconstruction, tutte_reconstruction, spring_err, tutte_err, springtime, tuttetime, tutte_reconstruction.face_enumeration_runtime

















################################ <> <> <> <> <> <> <> <> <> <> <> <> ################################

#The basic starter template - use these lines to get a basic reconstruction dataset
#only construct image, embeddings,
#no alignment, no same-delaunay comparison

# reload(parsift_lib)
# #basic run
# target_nsites = 500
# npolony = 100
# basic_circle = parsift_lib.Surface(target_nsites = target_nsites, directory_label='oop_testing')
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

