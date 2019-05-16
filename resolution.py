import networkx as nx

import math
import pytess
import litesim
import numpy as np
import sys
import ipdb
import matplotlib.pyplot as plt
import random
import os
import parsift_lib
from select import select
import execute_parsift
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
from scipy.stats import multivariate_normal
import scipy.stats as st
from execute_parsift import Experiment


def get_spot_polony(nspots,grid,corseed):#bc.site_coordinates,bc.corseed
    nsites=len(grid)
    nseed=len(corseed)
    sampleindx=random.sample(range(nsites),nspots)

    samplespot=[grid[i] for i in sampleindx]

    belong=np.zeros((nspots))

    for i in range(nspots):
        dis=np.zeros((nseed))
        for j in range(len(corseed)):
            dis[j]=(math.sqrt((samplespot[i][0]-corseed[j][0])**2+(samplespot[i][1]-corseed[j][1])**2))
        belong[i]=np.argmin(dis)


    return zip(samplespot,belong)


def get__fix_spot_polony(fixspotind,nseed,grid,corseed):#bc.site_coordinates,bc.corseed

    samplespot=[grid[i] for i in fixspotind]

    belong=np.zeros((len(fixspotind)))

    for i in range(len(fixspotind)):
        dis=np.zeros((nseed))
        for j in range(len(corseed)):
            dis[j]=(math.sqrt((samplespot[i][0]-corseed[j][0])**2+(samplespot[i][1]-corseed[j][1])**2))
        belong[i]=np.argmin(dis)


    return zip(samplespot,belong)


def get__cord_spot_polony(cord,nseed,corseed):



    belong=np.zeros((len(cord)))

    for i in range(len(cord)):
        dis=np.zeros((nseed))
        for j in range(len(corseed)):
            dis[j]=(math.sqrt((cord[i][0]-corseed[j][0])**2+(cord[i][1]-corseed[j][1])**2))
        belong[i]=np.argmin(dis)


    return zip(cord,belong)


def get_vor_centroid(vor):  # pytess.voronoi
    centroid = list()
    for i in vor:
        try:
            if len(i[0])>0:

                x_list = [vertex[0] for vertex in i[1]]
                y_list = [vertex[1] for vertex in i[1]]
                nvertex = len(i[1])
                x = sum(x_list) / nvertex
                y = sum(y_list) / nvertex
                centroid.append([x, y])
        except:
            print ('None')
    return centroid
class track_spot_centroid:
    def __init__(self,spots,seed,sprpos,tuttepos):#spots-polony,seed,sr.reconstructed_pos,tr.reconstructed.pos
        self.spots=spots
        self.seed=seed
        self.ori_vor=pytess.voronoi(seed)
        self.tutte_vor=pytess.voronoi(tuttepos)

        self.sp_vor=pytess.voronoi(sprpos)

    def tutte_centroid_error(self):
        ori_centroid=get_vor_centroid(self.ori_vor)
        self.ori_centroid=ori_centroid
        new_centroid=get_vor_centroid(self.tutte_vor)
        spot_ori=[ori_centroid[int(i[1])] for i in self.spots]
        self.spot_exact=[i[0] for i in self.spots]
        print self.spots
        spot_tutte=[new_centroid[int(i[1])]for i in self.spots]
        error=list()
        for i in range(len(spot_ori)):
            print i
            error.append( np.sqrt((spot_ori[i][0] - spot_tutte[i][0]) ** 2 + (spot_ori[i][1] - spot_tutte[i][1]) ** 2))
        error=[e/sum(error) for e in error]
        return error,spot_tutte
    def spr_centroid_error(self):
        ori_centroid=get_vor_centroid(self.ori_vor)
        self.ori_centroid=ori_centroid
        new_centroid=get_vor_centroid(self.sp_vor)
        spot_ori = [ori_centroid[int(i[1])] for i in self.spots]
        spot_spr = [new_centroid[int(i[1])] for i in self.spots]
        error=list()
        for i in range(len(spot_ori)):
            print i
            error.append( np.sqrt((spot_ori[i][0] - spot_spr[i][0]) ** 2 + (spot_ori[i][1] - spot_spr[i][1]) ** 2))
        error=[e/sum(error) for e in error]
        return error,spot_spr

def make_cross(grid,nl):
    nspot=nl*4+1

    x=grid[:,0]
    y=grid[:,1]
    left=np.argmin(x)
    right=np.argmax(x)

    up=np.argmax(y)
    down=np.argmin(y)
    left=grid[left]
    right=grid[right]
    up=grid[up]
    down=grid[down]
    xstep=(right[0]-left[0])/(2*nl)
    ystep=(up[1]-down[1])/(2*nl)

    center=[(left[0]+right[0])/2,(up[1]+down[1])/2]
    cross=np.zeros((nspot,2))
    cross[0]=1
    for i in range(nl):
        cross[i*4+1]=[left[0]+xstep*(i),left[1]]
        cross[i*4+2]=[right[0]-xstep*(i),right[1]]
        cross[i*4+3]=[down[0],down[1]+ystep*(i)]
        cross[i*4+4]=[up[0],up[1]-ystep*(i)]

    return cross

class resolution_experiment:
    def __init__(self,nspots):
        self.nspots=nspots


    def polony_number_variation_experiment(self):

        min = 10
        max = 50
        step = 10
        self.title = 'polony_variation'
        repeats_per_step = 3

        reload(litesim)
        self.master_directory = parsift_lib.prefix(self.title)

        number_of_steps = (max - min) / step

        master_errors_spring = np.zeros((number_of_steps * repeats_per_step, self.nspots))
        master_spring_centroid=np.zeros((number_of_steps * repeats_per_step, self.nspots,2))
        master_errors_tutte = np.zeros((number_of_steps * repeats_per_step, self.nspots))
        master_tutte_centroid=np.zeros((number_of_steps * repeats_per_step, self.nspots,2))
        polony_counts = np.zeros((number_of_steps * repeats_per_step))
        spot_ori_exact=np.zeros((number_of_steps * repeats_per_step, self.nspots,2))
        #fig_tutte=plt.figure()
        #fig_spr=plt.figure()
        #ax_tutte = fig_tutte.add_subplot(111, projection='3d')
        #ax_spr=fig_spr.add_subplot(111,projection='3d')
        fixspots=random.sample(range(295),50)
        fix_errors_spring = np.zeros((number_of_steps * repeats_per_step, 50))
        fix_spring_centroid=np.zeros((number_of_steps * repeats_per_step, 50,2))
        fix_errors_tutte = np.zeros((number_of_steps * repeats_per_step, 50))
        fix_tutte_centroid=np.zeros((number_of_steps * repeats_per_step, 50,2))
        fix_spot_ori_exact=np.zeros((number_of_steps * repeats_per_step, 50,2))
        for i in range(0, number_of_steps * repeats_per_step):

            npolony = min + i / repeats_per_step * step
            # substep = i%3
            print npolony
            polony_counts[i] = npolony
            #color=[random.random(),random.random(),random.random()]
            # master_errors_spring[i], master_errors_tutte[i]
            bc, sr, tr, se, te = execute_parsift.minimal_reconstruction_oop(target_nsites=5000, npolony=npolony,
                                                                            randomseed=i % repeats_per_step,
                                                                            filename='monalisacolor.png',
                                                                            master_directory=self.master_directory,
                                                                            iterlabel=str(npolony))
            samplespot=get_spot_polony(self.nspots,bc.nseed,bc.nsites,bc.site_coordinates,bc.corseed)
            spot_ori_exact[i]=np.array([q[0] for q in samplespot])
            #ax_tutte.scatter(spot_ori_exact[:,0],spot_ori_exact[:,1],np.repeat(npolony,self.nspots),marker='o',color=color)
            #ax_spr.scatter(spot_ori_exact[:,0],spot_ori_exact[:,1],np.repeat(npolony,self.nspots),marker='o',color=color)


            exp_resolution = track_spot_centroid(samplespot, bc.corseed, sr.reconstructed_points, tr.reconstructed_points)




            master_errors_tutte[i] = np.transpose(exp_resolution.tutte_centroid_error()[0])
            master_tutte_centroid[i]=exp_resolution.tutte_centroid_error()[1]
            master_errors_spring[i] = np.transpose(exp_resolution.spr_centroid_error()[0])
            master_spring_centroid[i]=exp_resolution.spr_centroid_error()[1]

            fixcorspotcentroid= get__fix_spot_polony(fixspots, bc.nseed, bc.site_coordinates, bc.corseed)

            fix_exp_resolution = track_spot_centroid(fixcorspotcentroid, bc.corseed, sr.reconstructed_points,
                                            tr.reconstructed_points)
            fix_spot_ori_exact[i]=np.array([q[0] for q in fixcorspotcentroid])
            fix_tutte_centroid[i]=fix_exp_resolution.tutte_centroid_error()[1]
            fix_errors_tutte[i] = np.transpose(fix_exp_resolution.tutte_centroid_error()[0])

            fix_spring_centroid[i]=fix_exp_resolution.spr_centroid_error()[1]
            fix_errors_spring[i] = np.transpose(fix_exp_resolution.tutte_centroid_error()[0])

            #ax_spr.scatter(master_spring_centroid[i,:,0],master_spring_centroid[i,:,1],np.repeat(npolony,self.nspots),color=color)
            #ax_tutte.scatter(master_tutte_centroid[i,:,0],master_tutte_centroid[i,:,1],np.repeat(npolony,self.nspots),color=color)
        self.polony_counts = polony_counts
        self.spring_centroid = master_spring_centroid
        self.tutte_centroid = master_tutte_centroid
        self.corspot=spot_ori_exact
        self.fix_spring_centroid=fix_spring_centroid
        self.fix_tutte_centroid=fix_tutte_centroid
        self.fix_corspot=fix_spot_ori_exact
        np.save(self.master_directory + '/' + 'corspot' + self.title + '.txt',zip(polony_counts, spot_ori_exact))

        np.save(self.master_directory + '/' + 'mastererrorsspring' + self.title + '.txt',zip(polony_counts, master_errors_spring))
        np.save(self.master_directory + '/' + 'centroidspring' + self.title + '.txt',zip(polony_counts, master_spring_centroid))

        np.save(self.master_directory + '/' + 'mastererrorstutte' + self.title + '.txt',zip(polony_counts, master_errors_tutte))
        np.save(self.master_directory + '/' + 'centroidtutte' + self.title + '.txt',zip(polony_counts, master_tutte_centroid))
        #fig_spr.savefig(master_directory + '/' + 'springspots' + title + '.png')
        #fig_tutte.savefig(master_directory + '/' + 'tuttespots' + title + '.png')
        timeout = 3
        print 'cd to ' + self.master_directory + '? (Y/n)'
        rlist, _, _ = select([sys.stdin], [], [], timeout)
        if rlist:
            cd_query = sys.stdin.readline()
            # ipdb.set_trace()
            print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
            if cd_query == 'y\n' or cd_query == 'yes\n' or cd_query == 'Y\n' or cd_query == 'YES\n' or cd_query == 'Yes\n' or cd_query == '\n':
                os.chdir(self.master_directory)
            else:
                print 'no cd'
        else:
            print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
            os.chdir(self.master_directory)

    def plot_in_3d(self):
        dspring = zip(self.polony_counts, self.tutte_centroid)
        dtutte = zip(self.polony_counts, self.spring_centroid)
        dspot = zip(self.polony_counts, self.corspot)
        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        for i in range(len(dspot)):
            color = [random.random(), random.random(), random.random()]

            ax.scatter(dspot[i][1][:, 0], dspot[i][1][:, 1], np.repeat(dspot[i][0], self.nspots), color=color,
                       marker='v')

            ax.scatter(dtutte[i][1][:, 0], dtutte[i][1][:, 1], np.repeat(dspot[i][0], self.nspots), color=color)

        plt.title('Spring reconstruction')

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        plt.title('Tutte reoconstruction')
        for i in range(len(dspot)):
            color = [random.random(), random.random(), random.random()]
            ax.scatter(dspot[i][1][:, 0], dspot[i][1][:, 1], np.repeat(dspot[i][0], self.nspots), color=color,
                       marker='^')

            ax.scatter(dspring[i][1][:, 0], dspring[i][1][:, 1], np.repeat(dspot[i][0], self.nspots), color=color)
        plt.show()
        fig.savefig(self.master_directory + '/' 'combine' + self.title + '.png')
    def fix_plot_in_3d(self):
        dspring = zip(self.polony_counts, self.fix_tutte_centroid)
        dtutte = zip(self.polony_counts, self.fix_spring_centroid)
        dspot = zip(self.polony_counts, self.fix_corspot)
        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        for i in range(len(dspot)):
            color = [random.random(), random.random(), random.random()]
            tutterescale = LA.norm(self.corspot[i]) / LA.norm(self.fix_tutte_centroid[i])
            ax.scatter(dspot[i][1][:, 0], dspot[i][1][:, 1], np.repeat(dspot[i][0], 50), color=color,
                       marker='v')

            ax.scatter(tutterescale**2*dtutte[i][1][:, 0], tutterescale**2*dtutte[i][1][:, 1], np.repeat(dspot[i][0], 50), color=color)

        plt.title('Tutte reconstruction')

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        plt.title('Spring reoconstruction')
        for i in range(len(dspot)):
            color = [random.random(), random.random(), random.random()]
            ax.scatter(dspot[i][1][:, 0], dspot[i][1][:, 1], np.repeat(dspot[i][0], 50), color=color,
                       marker='^')
            springrescale = LA.norm(self.corspot[i]) / LA.norm(self.fix_spring_centroid[i])

            ax.scatter(springrescale**2*dspring[i][1][:, 0], springrescale**2*dspring[i][1][:, 1], np.repeat(dspot[i][0], 50), color=color)
        plt.show()
        fig.savefig('fixcombine' + self.title + '.png')

def kernel_fit_plot(reconstructed_spots,fixedspor):

    for i in range(len(fixedspor)):
        fig = plt.figure(figsize=(13, 7))
        ax = plt.axes(projection='3d')
        print reconstructed_spots[:,i,0]
        valuex=reconstructed_spots[:,i,0]
        print valuex
        print reconstructed_spots[:,i,1]
        valuey=reconstructed_spots[:,i,1]

        values=np.stack([valuex,valuey])
        x=fixedspor[i][0]
        y=fixedspor[i][1]
        xx,yy=np.mgrid[(x-20):(x+20):100j,(y-20):(y+20):100j]
        plotset=np.vstack([xx.ravel(),yy.ravel()])
        kernel=st.gaussian_kde(values)
        z=np.reshape(kernel(plotset).T,xx.shape)

        surf = ax.plot_surface(xx, yy, z, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_zlabel('PDF')
        ax.set_title('Surface plot of Gaussian 2D KDE')
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.savefig('pdf'+'%s'%i+'.png')



def kernel_fit_plot_together(reconstructed_spots,fixedspor):
    fig = plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')
    for i in range(len(fixedspor)):

        print reconstructed_spots[:,i,0]
        valuex=reconstructed_spots[:,i,0]
        print valuex
        print reconstructed_spots[:,i,1]
        valuey=reconstructed_spots[:,i,1]

        values=np.stack([valuex,valuey])
        x=fixedspor[i][0]
        y=fixedspor[i][1]
        xx,yy=np.mgrid[(x-20):(x+20):10j,(y-20):(y+20):10j]
        plotset=np.vstack([xx.ravel(),yy.ravel()])
        kernel=st.gaussian_kde(values)
        z=np.reshape(kernel(plotset).T,xx.shape)

        surf = ax.plot_surface(xx, yy, z, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.set_zlabel('PDF')
    ax.set_title('Surface plot of Gaussian 2D KDE')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig('alltogetherpdf'+'%s'%i+'.png')



def kernel_fit_plot_percond(reconstructed_spots,fixedspor,nexp,nrepeat):

    for i in range(len(fixedspor)):
        for j in range(nexp):
            fig = plt.figure(figsize=(13, 7))
            ax = plt.axes(projection='3d')
            print reconstructed_spots[range(j*nrepeat,j*nrepeat+nrepeat), i, 0]
            valuex = reconstructed_spots[range(j*nrepeat,j*nrepeat+nrepeat), i, 0]
            print valuex
            print reconstructed_spots[range(j*nrepeat,j*nrepeat+nrepeat), i, 1]
            valuey = reconstructed_spots[range(j*nrepeat,j*nrepeat+nrepeat), i, 1]

            values = np.stack([valuex, valuey])
            x = fixedspor[i][0]
            y = fixedspor[i][1]
            xx, yy = np.mgrid[(x - 20):(x + 20):100j, (y - 20):(y + 20):100j]
            plotset = np.vstack([xx.ravel(), yy.ravel()])
            kernel = st.gaussian_kde(values)
            z = np.reshape(kernel(plotset).T, xx.shape)

            surf = ax.plot_surface(xx, yy, z, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
            ax.set_xlabel('x')
            ax.set_ylabel('y')

            ax.set_zlabel('PDF')
            ax.set_title('Surface plot of Gaussian 2D KDE')
            fig.colorbar(surf, shrink=0.5, aspect=5)

            plt.savefig('pdf_point' + '%s' % i+'cond'+'%s'%j + '.png')

def kernel_fit_plot_percond_together(reconstructed_spots,fixedspor,nexp,nrepeat):

    for i in range(nexp):
        fig = plt.figure(figsize=(13, 7))
        ax = plt.axes(projection='3d')
        xx, yy = np.mgrid[(- 20):(20):100j, (-20):(20):100j]
        for j in range(len(fixedspor)):

            print reconstructed_spots[range(i*nrepeat,i*nrepeat+nrepeat), j, 0]
            valuex = reconstructed_spots[range(i*nrepeat,i*nrepeat+nrepeat), j, 0]
            print valuex
            print reconstructed_spots[range(i*nrepeat,i*nrepeat+nrepeat), j, 1]
            valuey = reconstructed_spots[range(i*nrepeat,i*nrepeat+nrepeat), j, 1]

            values = np.stack([valuex, valuey])


            plotset = np.vstack([xx.ravel(), yy.ravel()])
            kernel = st.gaussian_kde(values)
            z = np.reshape(kernel(plotset).T, xx.shape)


            surf = ax.plot_surface(xx, yy, z, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_zlabel('PDF')

        ax.set_title('Surface plot of Gaussian 2D KDE')
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.savefig('pdf' + '_all_' +'cond'+'%s'%i + '.png')

def kernel_fit_plot_percond_one(reconstructed_spots,fixedspor,nexp,nrepeat):
    fig = plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')
    xx, yy = np.mgrid[(- 20):(20):100j, (-20):(20):100j]
    for i in range(nexp):

        for j in range(len(fixedspor)):

            print reconstructed_spots[range(i*nrepeat,i*nrepeat+nrepeat), j, 0]
            valuex = reconstructed_spots[range(i*nrepeat,i*nrepeat+nrepeat), j, 0]
            print valuex
            print reconstructed_spots[range(i*nrepeat,i*nrepeat+nrepeat), j, 1]
            valuey = reconstructed_spots[range(i*nrepeat,i*nrepeat+nrepeat), j, 1]

            values = np.stack([valuex, valuey])


            plotset = np.vstack([xx.ravel(), yy.ravel()])
            kernel = st.gaussian_kde(values)
            z = np.reshape(kernel(plotset).T, xx.shape)
            z+=(i+1)*10

            surf = ax.plot_surface(xx, yy, z, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.set_zlabel('PDF')

    ax.set_title('Surface plot of Gaussian 2D KDE')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig('pdf' + '_all_' +'cond'+'%s'%i + '.png')

class polony_number_resolution:
    def __init__(self,min=10,max=750,step=50,repeat=1,nspot=10,nl=2):
        self.min=min
        self.max=max
        self.nspot=nspot
        self.step=step
        self.repeat=repeat
        self.cross_layer=nl
    def polony_number_variation_experiment(self):
        min = self.min
        max =self.max
        step = self.step
        title = 'polony_variation'
        repeats_per_step = self.repeat

        reload(litesim)
        master_directory = parsift_lib.prefix(title)

        number_of_steps = (max - min) / step
        polony_counts = np.zeros((number_of_steps * repeats_per_step))
        master_spring_centroid = np.zeros((number_of_steps * repeats_per_step, self.nspot, 2))
        master_tutte_centroid = np.zeros((number_of_steps * repeats_per_step, self.nspot, 2))
        spot_ori_exact=np.zeros((number_of_steps * repeats_per_step, self.nspot,2))

        cross_tutte_centroid=np.zeros((number_of_steps * repeats_per_step, self.cross_layer*4+1, 2))
        cross_spring_centroid=np.zeros((number_of_steps * repeats_per_step, self.cross_layer*4+1, 2))
        cross_ori=np.zeros((number_of_steps * repeats_per_step, self.cross_layer*4+1, 2))

        peripheralpca_run_times = np.zeros((number_of_steps * repeats_per_step))
        totalpca_run_times = np.zeros((number_of_steps * repeats_per_step))
        spring_run_times = np.zeros((number_of_steps * repeats_per_step))
        tutte_run_times = np.zeros((number_of_steps * repeats_per_step))
        face_enumeration_run_times = np.zeros((number_of_steps * repeats_per_step))

        for i in range(0, number_of_steps * repeats_per_step):

            npolony = min + i / repeats_per_step * step
            nsites = npolony * 24
                # substep = i%3
            print npolony
            polony_counts[i] = npolony
                # master_errors_spring[i], master_errors_tutte[i]
                # basic_circle, spring_reconstruction, tutte_reconstruction, master_errors_spring[i], master_errors_tutte[
                #     i], spring_run_times[i], tutte_run_times[i],face_enumeration_run_times[i] = minimal_reconstruction_oop(target_nsites=nsites, npolony=npolony, randomseed=i % repeats_per_step,
                #                                     filename='monalisacolor.png', master_directory=master_directory,
                #                                     iterlabel=str(npolony))

            try:
                basic_run = Experiment(directory_label='single')
                basic_run.reconstruct(target_nsites=nsites, npolony=npolony, randomseed=i % repeats_per_step,
                                      filename='monalisacolor.png', master_directory=master_directory,
                                      iterlabel=str(npolony), full_output=False, do_peripheral_pca=False,
                                      do_total_pca=False,
                                      do_spring=True, do_tutte=True)


            except:
                print 'ERROR'
                # ipdb.set_trace()
                pass

            samplespot = get_spot_polony(self.nspot, basic_run.basic_circle.site_coordinates,basic_run.basic_circle.corseed)
            spot_ori_exact[i] = np.array([q[0] for q in samplespot])


            exp_resolution = track_spot_centroid(samplespot,basic_run.basic_circle.corseed, basic_run.spring_reconstruction.reconstructed_points,basic_run.tutte_reconstruction.reconstructed_points)
            master_tutte_centroid[i] = exp_resolution.tutte_centroid_error()[1]
            master_spring_centroid[i] = exp_resolution.spr_centroid_error()[1]

            mycross=make_cross(basic_run.basic_circle.site_coordinates,self.cross_layer)
            cross_ori[i]=mycross
            crosspol=track_spot_centroid(mycross,basic_run.basic_circle.corseed, basic_run.spring_reconstruction.reconstructed_points,basic_run.tutte_reconstruction.reconstructed_points)
            cross_tutte_centroid[i]=crosspol.tutte_centroid_error()[1]
            cross_spring_centroid[i]=crosspol.spr_centroid_error()[1]
                # ipdb.set_trace()



            np.savetxt(master_directory + '/' + '1peripheralpca_runtimes_' + title + '.txt',
                           zip(polony_counts, peripheralpca_run_times))
            np.savetxt(master_directory + '/' + '1totalpca_runtimes_' + title + '.txt',
                           zip(polony_counts, totalpca_run_times))

            np.savetxt(master_directory + '/' + '1spring_runtimes_' + title + '.txt',
                           zip(polony_counts, spring_run_times))
            np.savetxt(master_directory + '/' + '1tutte_runtimes_' + title + '.txt',
                           zip(polony_counts, tutte_run_times))

            np.savetxt(master_directory + '/' + '1face_enumeration_runtimes_' + title + '.txt',
                           zip(polony_counts, face_enumeration_run_times))


        self.polony_counts = polony_counts
        self.spring_centroid = master_spring_centroid
        self.tutte_centroid = master_tutte_centroid
        self.corspot=spot_ori_exact
        self.crossspot=cross_ori
        self.tutte_cross=cross_tutte_centroid
        self.spring_cross=cross_spring_centroid

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

    def plot_in_3d(self):
            dspring = zip(self.polony_counts, self.tutte_centroid)
            dtutte = zip(self.polony_counts, self.spring_centroid)
            dspot = zip(self.polony_counts, self.corspot)
            fig = plt.figure(figsize=plt.figaspect(0.5))
            gridscale=[max(self.polony_counts)/i for i in self.polony_counts]

            ax = fig.add_subplot(1, 2, 1, projection='3d')
            for i in range(len(dspot)):
                color = [random.random(), random.random(), random.random()]
                tutterescale = LA.norm(self.corspot[i]) / LA.norm(self.tutte_centroid[i])
                ax.scatter(math.sqrt(gridscale[i])*dspot[i][1][:, 0], math.sqrt(gridscale[i])*dspot[i][1][:, 1], np.repeat(dspot[i][0], self.nspot), color=color,
                           marker='v')

                ax.scatter(math.sqrt(gridscale[i])*tutterescale*dtutte[i][1][:, 0],math.sqrt(gridscale[i])*tutterescale*dtutte[i][1][:, 1], np.repeat(dspot[i][0], self.nspot), color=color)

            plt.title('Tutte reconstruction')

            ax = fig.add_subplot(1, 2, 2, projection='3d')
            plt.title('Spring reoconstruction')
            for i in range(len(dspot)):
                color = [random.random(), random.random(), random.random()]
                ax.scatter(math.sqrt(gridscale[i])*dspot[i][1][:, 0], math.sqrt(gridscale[i])*dspot[i][1][:, 1], np.repeat(dspot[i][0], self.nspot), color=color,
                           marker='^')
                springrescale = LA.norm(self.corspot[i]) / LA.norm(self.spring_centroid[i])
                ax.scatter(math.sqrt(gridscale[i])*springrescale*dspring[i][1][:, 0], springrescale*math.sqrt(gridscale[i])*dspring[i][1][:, 1], np.repeat(dspot[i][0], self.nspot), color=color)
            plt.show()
            fig.savefig('samplespot.png')

    def plot_cross_in_3d(self):
            dspring = zip(self.polony_counts, self.spring_cross)
            dtutte = zip(self.polony_counts, self.tutte_cross)
            dspot = zip(self.polony_counts, self.crossspot)
            fig = plt.figure(figsize=plt.figaspect(0.5))
            gridscale=[max(self.polony_counts)/i for i in self.polony_counts]
            npoint=self.cross_layer*4+1
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            for i in range(len(dspot)):
                color = [random.random(), random.random(), random.random()]
                tutterescale = LA.norm(self.corspot[i]) / LA.norm(self.tutte_cross[i])
                ax.scatter(math.sqrt(gridscale[i])*dspot[i][1][:, 0], math.sqrt(gridscale[i])*dspot[i][1][:, 1], np.repeat(dspot[i][0], npoint), color=color,
                           marker='v')

                ax.scatter(math.sqrt(gridscale[i])*tutterescale*dtutte[i][1][:, 0],math.sqrt(gridscale[i])*tutterescale*dtutte[i][1][:, 1], np.repeat(dspot[i][0], npoint), color=color)

            plt.title('Tutte reconstruction')

            ax = fig.add_subplot(1, 2, 2, projection='3d')
            plt.title('Spring reoconstruction')
            for i in range(len(dspot)):
                color = [random.random(), random.random(), random.random()]
                ax.scatter(math.sqrt(gridscale[i])*dspot[i][1][:, 0], math.sqrt(gridscale[i])*dspot[i][1][:, 1], np.repeat(dspot[i][0], npoint), color=color,
                           marker='^')
                springrescale = LA.norm(self.corspot[i]) / LA.norm(self.spring_cross[i])
                ax.scatter(math.sqrt(gridscale[i])*springrescale*dspring[i][1][:, 0], springrescale*math.sqrt(gridscale[i])*dspring[i][1][:, 1], np.repeat(dspot[i][0], npoint), color=color)
            plt.show()
            fig.savefig('cross.png')

def make_cross(grid,nl):
    nspot=nl*4+1

    x=grid[:,0]
    y=grid[:,1]
    left=np.argmin(x)
    right=np.argmax(x)

    up=np.argmax(y)
    down=np.argmin(y)
    left=grid[left]
    right=grid[right]
    up=grid[up]
    down=grid[down]
    xstep=(right[0]-left[0])/(2*nl)
    ystep=(up[1]-down[1])/(2*nl)

    center=[(left[0]+right[0])/2,(up[1]+down[1])/2]
    cross=np.zeros((nspot,2))
    cross[0]=1
    for i in range(nl):
        cross[i*4+1]=[left[0]+xstep*(i),left[1]]
        cross[i*4+2]=[right[0]-xstep*(i),right[1]]
        cross[i*4+3]=[down[0],down[1]+ystep*(i)]
        cross[i*4+4]=[up[0],up[1]-ystep*(i)]

    return cross



