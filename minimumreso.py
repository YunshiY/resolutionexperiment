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
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.mlab as mlab
import seaborn as sns
import pandas as pd
from scipy.signal import argrelextrema,argrelmax,argrelmin

from execute_parsift import Experiment
import statsmodels.api as sm


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
def get_point_polony(points,corseed):#bc.site_coordinates,bc.corseed


    belong=np.zeros((len(points)))

    for i in range(len(points)):
        dis=np.zeros((len(corseed)))
        for j in range(len(corseed)):
            dis[j]=(math.sqrt((points[i][0]-corseed[j][0])**2+(points[i][1]-corseed[j][1])**2))
        belong[i]=np.argmin(dis)


    return zip(points,belong)

def get__cord_spot_polony(cord,nseed,corseed):



    belong=np.zeros((len(cord)))

    for i in range(len(cord)):
        dis=np.zeros((nseed))
        for j in range(len(corseed)):
            dis[j]=(math.sqrt((cord[i][0]-corseed[j][0])**2+(cord[i][1]-corseed[j][1])**2))
        belong[i]=np.argmin(dis)


    return zip(cord,belong)


def get_vor_centroid(vor):  # pytess.voronoi
    centroid = []
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
        print len(new_centroid)
        spot_ori=[ori_centroid[int(i[1])] for i in self.spots]
        self.spot_exact=[i[0] for i in self.spots]
        spot_tutte=[]
        for i in self.spots:
            try:
                spot_tutte.append(new_centroid[int(i[1])])
            except:
                spot_tutte.append(random.choice(range(len(new_centroid))))
        error=[]
        for i in range(len(spot_ori)):
            try:
                error.append( np.sqrt((spot_ori[i][0] - spot_tutte[i][0]) ** 2 + (spot_ori[i][1] - spot_tutte[i][1]) ** 2))
            except:
                error.append(None)
        return error,spot_tutte
    def spr_centroid_error(self):
        ori_centroid=get_vor_centroid(self.ori_vor)
        self.ori_centroid=ori_centroid
        new_centroid=get_vor_centroid(self.sp_vor)
        print len(new_centroid)
        spot_ori = [ori_centroid[int(i[1])] for i in self.spots]
        spot_spring = []
        for i in self.spots:
            try:
                spot_spring.append(new_centroid[int(i[1])])
            except:
                spot_spring.append(random.choice(range(len(new_centroid))))
        error=[]
        for i in range(len(spot_ori)):
            try:
                error.append( np.sqrt((spot_ori[i][0] - spot_spring[i][0]) ** 2 + (spot_ori[i][1] - spot_spring[i][1]) ** 2))
            except:
                error.append(None)

        return error,spot_spring

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
            plt.close()
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
    plt.close()

class just_generate_data:

    def experiment(self,nsites,npolony,directory_label=None,master_directory=None,iterlabel=None,full_output=False,filename='monalisacolor.png'):
        if not master_directory:
            directory_name = parsift_lib.prefix(directory_label)
        else:
            directory_name = master_directory
        self.basic_circle=parsift_lib.Surface(target_nsites=nsites, directory_label='',
                                           master_directory=directory_name)
        self.basic_circle.circular_grid_ian()
        self.basic_circle.scale_image_axes(filename='monalisacolor.png')

        self.basic_circle.seed(nseed=npolony,full_output=full_output)
        self.basic_circle.crosslink_polonies(full_output=full_output)
        self.spring_reconstruction=parsift_lib.Reconstruction(self.basic_circle.directory_name,
                                                               corseed=self.basic_circle.corseed,
                                                               ideal_graph=self.basic_circle.ideal_graph,
                                                               untethered_graph=self.basic_circle.untethered_graph,
                                                               rgb_stamp_catalog=self.basic_circle.rgb_stamp_catalog)
        self.spring_reconstruction.conduct_spring_embedding(full_output=full_output, image_only=False)
        self.spring_reconstruction.align(full_output=False)
        self.tutte_reconstruction=parsift_lib.Reconstruction(self.basic_circle.directory_name,
                                                               corseed=self.basic_circle.corseed,
                                                               ideal_graph=self.basic_circle.ideal_graph,
                                                               untethered_graph=self.basic_circle.untethered_graph,
                                                               rgb_stamp_catalog=self.basic_circle.rgb_stamp_catalog)

        self.tutte_reconstruction.conduct_tutte_embedding(full_output=full_output,image_only=False)
        self.tutte_reconstruction.align(full_output=False)



class polony_number_resolution_minimum:
    def __init__(self,min=10,max=5000,step=500,repeat=1,nspot=10,nl=2):
        self.min=min
        self.max=max
        self.nspot=nspot
        self.step=step
        self.repeat=repeat
        self.cross_layer=nl
    def polony_number_variation_experiment(self):
        print '###################%%%%%%%%%%%%%%%%%%%%%%%%%%%'
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
        seedlist=[]
        sitelist=[]
        springpolonylist=[]
        tuttepolonylist=[]
        spring_adjusted=[]
        tutte_adjusted=[]
        simulationerror=[]
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



            self.basic_run = just_generate_data()
            try:
                self.basic_run.experiment(nsites=nsites, npolony=npolony,
                                      filename='monalisacolor.png', master_directory=master_directory,
                                      iterlabel=str(npolony), full_output=False)



                seedlist.append(self.basic_run.basic_circle.corseed)
                sitelist.append(self.basic_run.basic_circle.site_coordinates)
            #except:
             #   print 'ERROR'
                # ipdb.set_trace()
              #  pass

                samplespot = get_spot_polony(self.nspot, self.basic_run.basic_circle.site_coordinates,self.basic_run.basic_circle.corseed)
                spot_ori_exact[i] = np.array([q[0] for q in samplespot])


                exp_resolution = track_spot_centroid(samplespot,self.basic_run.basic_circle.corseed, self.basic_run.spring_reconstruction.reconstructed_points,self.basic_run.tutte_reconstruction.reconstructed_points)
                master_tutte_centroid[i] = exp_resolution.tutte_centroid_error()[1]
                master_spring_centroid[i] = exp_resolution.spr_centroid_error()[1]

                mycross=make_cross(self.basic_run.basic_circle.site_coordinates,self.cross_layer)
                cross_ori[i]=mycross
                crosspol=track_spot_centroid(mycross,self.basic_run.basic_circle.corseed, self.basic_run.spring_reconstruction.reconstructed_points,self.basic_run.tutte_reconstruction.reconstructed_points)
                cross_tutte_centroid[i]=crosspol.tutte_centroid_error()[1]
                cross_spring_centroid[i]=crosspol.spr_centroid_error()[1]
                # ipdb.set_trace()
                springponolylist.append(self.basic_run.spring_reconstruction.reconstructed_points)
                tutteponolylist.append(self.basic_run.tutte_reconstruction.reconstructed_points)
                spring_adjusted.append(self.basic_run.spring_reconstruction.corseed_adjusted)
                tutte_adjusted.append(self.basic_run.tutte_reconstruction.corseed_adjusted
                )


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

            except:
                print 'simulation fail'
                simulationerror.append(i)
                try:
                    seedlist[i]=[0]
                except:
                    seedlist.append([0])
                try:
                    sitelist[i] = [0]
                except:
                    sitelist.append([0])
                springpolonylist.append([0])
                tuttepolonylist.append([0])



        self.springallpolony=springpolonylist
        self.tutteallpolony=tuttepolonylist
        self.polony_counts = polony_counts
        self.spring_centroid = master_spring_centroid
        self.tutte_centroid = master_tutte_centroid
        self.corspot=spot_ori_exact
        self.crossspot=cross_ori
        self.tutte_cross=cross_tutte_centroid
        self.spring_cross=cross_spring_centroid
        self.sitelist=sitelist
        self.seedlist=seedlist
        self.tutte_adjusted_seed=tutte_adjusted
        self.spring_adjusted_seed=spring_adjusted
        self.simulationerror=simulationerror
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

def kernel_error(reconstructed_spots,fixedspor,npoints,nexp,nrepeat):
    mat_error=reconstructed_spots-fixedspor

    for i in range(len(reconstructed_spots)):
        fig = plt.figure(figsize=(13, 7))
        ax = plt.axes(projection='3d')
        values=np.stack([mat_error[i][:,0],mat_error[i][:,1]])
        kernel=st.gaussian_kde(values)
        for j in range(npoints):
            x=fixedspor[j,0]
            y=fixedspor[j,1]
            xx, yy = np.mgrid[(x- 10):(x+10):100j, (y-10):( y+10):100j]
            plotset = np.vstack([xx.ravel(), yy.ravel()])
            z=np.reshape(kernel(plotset).T,xx.shape)

            surf = ax.plot_surface(xx, yy, z, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_zlabel('PDF')
        ax.set_title('Surface plot of Gaussian 2D KDE')
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.savefig('error_kernel' + '%s' % i + '.png')

        ipdb.set_trace()



    def plot_cross_in_3d(f):
            dspring = zip(f.polony_counts, f.spring_cross)
            dtutte = zip(f.polony_counts, f.tutte_cross)
            dspot = zip(f.polony_counts, f.crossspot)
            fig = plt.figure(figsize=plt.figaspect(0.5))
            gridscale=[max(f.polony_counts)/i for i in f.polony_counts]
            npoint=f.cross_layer*4+1
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            for i in range(len(dspot)):
                color = [random.random(), random.random(), random.random()]
                tutterescale = LA.norm(f.corspot[i]) / LA.norm(f.tutte_cross[i])

                ax.scatter(math.sqrt(gridscale[i])*tutterescale*dtutte[i][1][:, 0],math.sqrt(gridscale[i])*tutterescale*dtutte[i][1][:, 1], np.repeat(dspot[i][0], npoint), color=color)

            plt.title('Tutte reconstruction')

            ax = fig.add_subplot(1, 2, 2, projection='3d')
            plt.title('Spring reoconstruction')
            for i in range(len(dspot)):
                color = [random.random(), random.random(), random.random()]

                springrescale = LA.norm(f.corspot[i]) / LA.norm(f.spring_cross[i])
                ax.scatter(math.sqrt(gridscale[i])*springrescale*dspring[i][1][:, 0], springrescale*math.sqrt(gridscale[i])*dspring[i][1][:, 1], np.repeat(dspot[i][0], npoint), color=color)
            plt.show()
            fig.savefig('cross.png')

def extract_px_polony(exp,nl):
    tuttelist=list()
    springlist=list()
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    original=list()
    for i in range(len(exp.seedlist)):
        points = make_cross(exp.sitelist[i],nl

        )
        id=get_point_polony(points,exp.seedlist[i])
        original.append(points)
        print id
        estimation=track_spot_centroid(id,exp.seedlist[i],exp.springallpolony[i],exp.tutteallponoly[i])
        tutte_error,tutte_spot=estimation.tutte_centroid_error()
        print tutte_spot
        color = [random.random(), random.random(), random.random()]
        for spot in tutte_spot:
            ax.scatter (spot[0],
                    spot[1], exp.polony_counts[i],
                   color=color)

        plt.title('Tutte reconstruction')

    ax = fig.add_subplot(1, 2, 2, projection='3d')

    for i in range(len(exp.seedlist)):
        points = make_cross(exp.sitelist[i], nl

                                )

        id = get_point_polony(points, exp.seedlist[i])
        print id
        estimation = track_spot_centroid(id, exp.seedlist[i], exp.springallpolony[i], exp.tutteallponoly[i])
        spring_error, spring_spot = estimation.spr_centroid_error()
        plt.title('Spring reoconstruction')
        color = [random.random(), random.random(), random.random()]
        for spot in spring_spot:

            ax.scatter(spot[0],
                   spot[1], exp.polony_counts[i],
                   color=color)
    tuttelist.append(tutte_spot)
    springlist.append(spring_spot)
    plt.show()

    fig.savefig('cross.png')
    plt.close()

    return tuttelist,springlist

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


def extract_adjusted_px_polony(exp,nl):
    tuttelist=list()
    springlist=list()
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    original=list()
    for i in range(len(exp.seedlist)):
        points = make_cross(exp.sitelist[i],nl

        )
        id=get_point_polony(points,exp.seedlist[i])
        original.append(points)
        print id
        estimation=track_spot_centroid(id,exp.seedlist[i],exp.spring_adjusted_seed[i],exp.tutte_adjusted_seed[i])
        tutte_error,tutte_spot=estimation.tutte_centroid_error()
        print tutte_spot
        color = [random.random(), random.random(), random.random()]
        for spot in tutte_spot:
            ax.scatter (spot[0],
                    spot[1], exp.polony_counts[i],
                   color=color)

    plt.title('Tutte reconstruction')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('polony number')
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    for i in range(len(exp.seedlist)):
        points = make_cross(exp.sitelist[i], nl)
        id = get_point_polony(points, exp.seedlist[i])
        print id
        estimation = track_spot_centroid(id, exp.seedlist[i], exp.spring_adjusted_seed[i], exp.tutte_adjusted_seed[i])
        spring_error, spring_spot = estimation.spr_centroid_error()
        color = [random.random(), random.random(), random.random()]
        for spot in spring_spot:

            ax.scatter(spot[0],
                   spot[1], exp.polony_counts[i],
                   color=color)
        tuttelist.append(tutte_spot)
        springlist.append(spring_spot)
    plt.title('Spring reoconstruction')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('polony number')

    plt.show()

    fig.savefig('cross.png')
    plt.close()

    return tuttelist,springlist,original
def plot_normal_error_scatter(exp):
    normal_factor=np.zeros((len(exp.seedlist)))
    for i in range(len(exp.seedlist)):
        normal_factor[i]=len(exp.seedlist[-1])/len(exp.seedlist[i])

    errormat_spring=list()

    for i in range(len(exp.spring_adjusted_seed)):
        errormat_spring.append(normal_factor[i]*(exp.spring_adjusted_seed[i]-exp.seedlist[i]))
    errormat_tutte = list()
    for i in range(len(exp.tutte_adjusted_seed)):
        errormat_tutte.append(normal_factor[i]*(exp.tutte_adjusted_seed[i] - exp.seedlist[i]))

    fig=plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    for i in range(len(errormat_spring)):
        color = [random.random(), random.random(), random.random()]

        for spot in errormat_spring[i]:

            ax.scatter(spot[0],
                   spot[1], exp.polony_counts[i],
                   color=color)
    plt.title('Tutte reconstruction')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.set_zlabel('polony number')
    ax = fig.add_subplot(1, 2, 2, projection='3d')

    for i in range(len(errormat_tutte)):
        color = [random.random(), random.random(), random.random()]

        for spot in errormat_tutte[i]:
            ax.scatter(spot[0],
                       spot[1], exp.polony_counts[i],
                       color=color)
    plt.title('Spring reoconstruction')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.set_zlabel('polony number')
    plt.savefig('normalized_grid_size.png')
    plt.close()
def plot_error_scatter(exp,unitseedlist):

    errormat_spring=[]

    for i in range(len(exp.spring_adjusted_seed)):
        errormat_spring.append(exp.spring_adjusted_seed[i]-unitseedlist[i])
    errormat_tutte = []
    for i in range(len(exp.tutte_adjusted_seed)):
        errormat_tutte.append(exp.tutte_adjusted_seed[i] - unitseedlist[i])

    fig=plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    for i in range(len(errormat_spring)):
        color = [random.random(), random.random(), random.random()]

        for spot in errormat_spring[i]:

            ax.scatter(spot[0],spot[1], exp.polony_counts[i],
                   color=color)
    plt.title('Tutte reconstruction')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.set_zlabel('polony number')
    ax = fig.add_subplot(1, 2, 2, projection='3d')

    for i in range(len(errormat_tutte)):
        color = [random.random(), random.random(), random.random()]

        for spot in errormat_tutte[i]:
            ax.scatter(spot[0],
                       spot[1], exp.polony_counts[i],
                       color=color)
    plt.title('Spring reoconstruction')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.set_zlabel('polony number')
    plt.savefig('grid_size_error_new.png')
    plt.close()



def kernel_error_fit_plot(seedlist,k):

    for i in range(len(seedlist)):
        fig = plt.figure(figsize=(13, 7))
        ax = plt.axes(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_zlabel('PDF')
        x=seedlist[i][0]
        y=seedlist[i][1]
        xx,yy=np.mgrid[(x-2):(x+2):100j,(y-2):(y+2):100j]
        plotset=np.vstack([xx.ravel(),yy.ravel()])

        z=np.reshape(k(plotset).T,xx.shape)

        surf = ax.plot_surface(xx, yy, z, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')

        ax.set_title('Surface plot of Gaussian 2D KDE')
        fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig('pdf_test.png')



def test_kernel_error_fit_plot(seedlist,k):
    xx, yy = np.mgrid[(- 2):(+ 2):100j, ( - 2):(+ 2):100j]
    plotset = np.vstack([xx.ravel(), yy.ravel()])
    for i in range(len(seedlist)):
        fig = plt.figure(figsize=(13, 7))
        ax = plt.axes(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_zlabel('PDF')
        x=seedlist[i][0]
        y=seedlist[i][1]


        z=np.reshape(k(plotset).T,xx.shape)

        surf = ax.plot_surface(xx+x, yy+y, z, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')

        ax.set_title('Surface plot of Gaussian 2D KDE')
        fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    plt.savefig('pdf_test.png')
    plt.close()
class normal_centroid_experiment:
    def __init__(self,exp):
        self.__dict__.update(exp.__dict__)
    def normalfactor(self):
        normalfactor=[]
        for i in range(len(self.polony_counts)):
            normalfactor.append(np.ceil(max(self.sitelist[i][:,0])))
        self.normalfactor=normalfactor
    def normal_coord(self):
        unit_seedlist = []
        for i in range(len(self.seedlist)):
            unit_seedlist.append(self.seedlist[i] / self.normalfactor[i])
        unit_sitelist = []
        for i in range(len(self.sitelist)):
            unit_sitelist.append(self.sitelist[i] / self.normalfactor[i])
        self.seedlist=unit_seedlist
        self.sitelist=unit_sitelist

    def get_pol_centroid(self):
        polcentroidlist = []
        for i in range(len(self.seedlist)):
            polcentroidlist.append(get_point_polony(self.seedlist[i], self.seedlist[i]))
        self.polcentroidlist = polcentroidlist



    def list_centroid_error(self):
        tutte_centroidlist = []
        tutte_error = []
        spring_error = []
        spring_centroidlist = []
        for i in range(len(self.seedlist)):
            track_centroid = track_spot_centroid(self.polcentroidlist[i], self.seedlist[i], self.spring_adjusted_seed[i],
                                                 self.tutte_adjusted_seed[i])
            temp_error_tutte, tutte_temporarylist = track_centroid.tutte_centroid_error()
            temp_error_spring, spring_temporarylist = track_centroid.spr_centroid_error()
            tutte_centroidlist.append(tutte_temporarylist)
            tutte_error.append(temp_error_tutte)
            spring_error.append(temp_error_spring)
            spring_centroidlist.append(spring_temporarylist)
        self.spring_centroidlist = spring_centroidlist
        self.spring_centroid_distance_error = spring_error
        self.tutte_centroid_distance_error = tutte_error
        self.tutte_centroidlist = tutte_centroidlist

    def normal_centroid_error_list(self):
        errorlist_spring = []

        for i in range(len(self.spring_centroidlist)):
            errorlist_spring.append(self.spring_centroidlist[i] - self.seedlist[i])
        errorlist_tutte = []
        for i in range(len(self.tutte_adjusted_seed)):
            errorlist_tutte.append(self.tutte_centroidlist[i] - self.seedlist[i])
        self.spring_centroid_corerror=errorlist_spring
        self.tutte_centroid_corerror=errorlist_tutte


    def kernel_per_experiment(self):
        springkernels = []
        tuttekernels = []
        for i in range(len(self.seedlist)):
            springkernels.append(st.gaussian_kde(np.stack([np.array(self.spring_centroidlist[i])[:, 0], np.array(self.spring_centroidlist[i])[:, 1]])))
        for i in range(len(self.seedlist)):
            tuttekernels.append(st.gaussian_kde(np.stack([np.array(self.tutte_centroidlist[i])[:, 0], np.array(self.tutte_centroidlist[i])[:, 1]])))
        self.spring_kernellist=springkernels
        self.tutte_kernellist=tuttekernels
        return springkernels, tuttekernels




    def test_kernel_list_error_fit_plot(self):
        xx, yy = np.mgrid[(- 2):(+ 2):100j, (- 2):(+ 2):100j]
        plotset = np.vstack([xx.ravel(), yy.ravel()])
        fig = plt.figure(figsize=(13, 7))

        ax = plt.axes(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_zlabel('PDF')
        ax.set_title('Tutte reconstruction Point distribution')
        counter=0
        for i in self.tutte_kernellist:
            z = np.reshape(i(plotset).T, xx.shape)
            z += counter

            surf = ax.plot_surface(xx, yy, z, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')

            x = fig.colorbar(surf, shrink=0.5, aspect=5)
            x.set_label('%s' % counter + 'polonies')
            counter+=1


        plt.savefig('tutte_single_point.png')
        plt.show()

        plt.close()
        counter=0
        for i in self.spring_kernellist:

            z = np.reshape(i(plotset).T, xx.shape)
            z += counter

            surf = ax.plot_surface(xx, yy, z, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')

            x = fig.colorbar(surf, shrink=0.5, aspect=5)
            x.set_label('%s' % counter + 'polonies')
            counter+=1

        plt.savefig('spring_single_point.png')
        plt.show()

        plt.close()


    def plot_centroid_error_scatter(self ):


        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        for i in range(len(self.spring_centroid_corerror)):
            color = [random.random(), random.random(), random.random()]

            for spot in self.tutte_centroid_corerror[i]:
                ax.scatter(spot[0],
                           spot[1], self.polony_counts[i],
                           color=color)
        plt.title('Tutte reconstruction')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_zlabel('polony number')
        ax = fig.add_subplot(1, 2, 2, projection='3d')

        for i in range(len(self.spring_centroid_corerror)):
            color = [random.random(), random.random(), random.random()]

            for spot in self.spring_centroid_corerror[i]:
                ax.scatter(spot[0],
                           spot[1], self.polony_counts[i],
                           color=color)
        plt.title('Spring reoconstruction')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_zlabel('polony number')
        plt.savefig('centroid_grid_size_error_new.png')
        plt.close()

    def his2d_experiment(self):
        tittle = 'hist2d'
        master_directory = parsift_lib.prefix(tittle)
        counter=0
        for i in self.tutte_centroid_corerror:
            i=np.array(i)
            plt.hist2d(i[:,0],i[:,1])
            counter+=1

            plt.savefig(master_directory + '/' + '%s' % counter + 'tutte.png')
            plt.close()
        counter=0
        for i in self.spring_centroid_corerror:
            i=np.array(i)
            plt.hist2d(i[:,0] ,
                       i[:,1])
            counter+=1
            plt.savefig(master_directory + '/' + '%s' % counter + 'spring.png')
            plt.close()

    def his1d_experiment(self):

        tittle = 'hist1d'
        master_directory = parsift_lib.prefix(tittle)



        for i in range(len(self.tutte_centroid_distance_error)):
            counter = self.polony_counts[i * self.repeat]
            sns.distplot(self.tutte_centroid_distance_error[i], norm_hist=True, kde=True,
                         kde_kws={'linewidth': 3},
                         label=('ponoly number'+'%s'%counter))
        plt.legend(prop={'size': 16}, title = 'Polony number')

        plt.title('density plot of distance error from tutte reconstruction')
        plt.savefig(master_directory + '/' + '%s' % i + 'tutte_distance_error.png')

        plt.close()
        for i in range(len(self.spring_centroid_distance_error)):
            counter = self.polony_counts[i * self.repeat]
            sns.distplot(self.spring_centroid_distance_error[i], norm_hist=True, kde=True,
                         kde_kws={'linewidth': 3},
                         label=('ponoly number' + '%s' % counter))
        plt.legend(prop={'size': 16}, title='Polony number')

        plt.title('density plot of distance error from spring reconstruction')
        plt.savefig(master_directory + '/' + '%s' % i + 'spring_distance_error.png')

        plt.close()




    def his1d_experiment_repeat(self):

        title = 'hist1d_repeat'
        master_directory = parsift_lib.prefix(title)
        tutte_merge_list=[]
        spring_merge_list=[]
        for i in range(int(len(self.seedlist)/self.repeat)):
            tutte_merge_list.append(np.concatenate(self.tutte_centroid_distance_error[(i*self.repeat):(i+1)*self.repeat],axis=None))
        for i in range(int(len(self.seedlist)/self.repeat)):
            spring_merge_list.append(np.concatenate(self.spring_centroid_distance_error[(i*self.repeat):(i+1)*self.repeat],axis=None))


        for i in range(len(tutte_merge_list)):
            counter=self.polony_counts[i*self.repeat]

            n,bins,patches=plt.hist(tutte_merge_list[i], alpha=0.5, label=('%s' % i),normed=True)
            mu,sigma=norm.fit(tutte_merge_list[i])
            y=mlab.normpdf(bins,mu,sigma)
            l=plt.plot(bins,y,'r--',linewidth=2)
            plt.title('point distribution of distance error from tutte reconstruction of '+'%s'%counter+'polonies'+'\n fit by gaussian distribution')
            plt.legend()
            plt.savefig(master_directory + '/' + '%s' % i + 'tutte.png')

            plt.close()
        for i in range(len(tutte_merge_list)):
            counter = self.polony_counts[i * self.repeat]
            sns.distplot(tutte_merge_list[i], norm_hist=True, kde=True,
                         kde_kws={'linewidth': 3},
                         label=('ponoly number'+'%s'%counter))

            plt.title(
                'distribution of distance error from tutte reconstruction of ' + '%s' % counter + 'polonies'+'\n fit by kernel density estimation')
            plt.savefig(master_directory + '/' + '%s' % i + 'kernel_tutte.png')

            plt.close()
        for i in range(len(spring_merge_list)):
            n,bins,pathces=plt.hist(spring_merge_list[i], alpha=0.5, label='%s' % i,normed=True)
            mu, sigma = norm.fit(spring_merge_list[i])
            y = mlab.normpdf(bins, mu, sigma)
            l = plt.plot(bins, y, 'r--', linewidth=2)
            plt.title('point distribution of distance error from spring reconstruction of '+'%s'%counter+'polonies'+'\n fit by guassian distribition')
            plt.savefig(master_directory + '/' + '%s' % i + 'spring.png')
            plt.close()
        for i in range(len(spring_merge_list)):
            counter = self.polony_counts[i * self.repeat]
            sns.distplot(spring_merge_list[i], norm_hist=True, kde=True,
                         kde_kws={'linewidth': 3},
                         label=('ponoly number'+'%s'%counter))

            plt.title(
                'distribution of distance error from spring reconstruction of ' + '%s' % counter + 'polonies'+'\n fit by kernel density estimation')
            plt.savefig(master_directory + '/' + '%s' % i + 'kernel_spring.png')

            plt.close()


def run_analysis_experiemnt(analyze_object):
    analyze_object.normalfactor()
    analyze_object.normal_coord()
    analyze_object.get_pol_centroid()
    analyze_object.list_centroid_error()
    analyze_object.normal_centroid_error_list()
    analyze_object.kernel_per_experiment()
    analyze_object.test_kernel_list_error_fit_plot()
    analyze_object.plot_centroid_error_scatter()
    analyze_object.his1d_experiment()
    analyze_object.his1d_experiment_repeat()

class centroid_experiment:
    def __init__(self,exp):
        self.__dict__.update(exp.__dict__)


    def get_pol_centroid(self):
        polcentroidlist = []
        for i in range(len(self.seedlist)):
            polcentroidlist.append(get_point_polony(self.seedlist[i], self.seedlist[i]))
        self.polcentroidlist = polcentroidlist



    def list_centroid_error(self):
        tutte_centroidlist = []
        tutte_error = []
        spring_error = []
        spring_centroidlist = []
        for i in range(len(self.seedlist)):
            track_centroid = track_spot_centroid(self.polcentroidlist[i], self.seedlist[i], self.spring_adjusted_seed[i],
                                                 self.tutte_adjusted_seed[i])
            temp_error_tutte, tutte_temporarylist = track_centroid.tutte_centroid_error()
            temp_error_spring, spring_temporarylist = track_centroid.spr_centroid_error()
            tutte_centroidlist.append(tutte_temporarylist)
            tutte_error.append(temp_error_tutte)
            spring_error.append(temp_error_spring)
            spring_centroidlist.append(spring_temporarylist)
        self.spring_centroidlist = spring_centroidlist
        self.spring_centroid_distance_error = spring_error
        self.tutte_centroid_distance_error = tutte_error
        self.tutte_centroidlist = tutte_centroidlist

    def normal_centroid_error_list(self):
        errorlist_spring = []

        for i in range(len(self.spring_centroidlist)):
            errorlist_spring.append(self.spring_centroidlist[i] - self.seedlist[i])
        errorlist_tutte = []
        for i in range(len(self.tutte_adjusted_seed)):
            errorlist_tutte.append(self.tutte_centroidlist[i] - self.seedlist[i])
        self.spring_centroid_corerror=errorlist_spring
        self.tutte_centroid_corerror=errorlist_tutte


    def kernel_per_experiment(self):
        springkernels = []
        tuttekernels = []
        for i in range(len(self.seedlist)):
            springkernels.append(st.gaussian_kde(np.stack([np.array(self.spring_centroidlist[i])[:, 0], np.array(self.spring_centroidlist[i])[:, 1]])))
        for i in range(len(self.seedlist)):
            tuttekernels.append(st.gaussian_kde(np.stack([np.array(self.tutte_centroidlist[i])[:, 0], np.array(self.tutte_centroidlist[i])[:, 1]])))
        self.spring_kernellist=springkernels
        self.tutte_kernellist=tuttekernels
        return springkernels, tuttekernels




    def test_kernel_list_error_fit_plot(self):
        xx, yy = np.mgrid[(- 2):(+ 2):100j, (- 2):(+ 2):100j]
        plotset = np.vstack([xx.ravel(), yy.ravel()])
        fig = plt.figure(figsize=(13, 7))

        ax = plt.axes(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_zlabel('PDF')
        ax.set_title('Tutte reconstruction Point distribution')
        counter=0
        for i in self.tutte_kernellist:
            z = np.reshape(i(plotset).T, xx.shape)
            z += counter

            surf = ax.plot_surface(xx, yy, z, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')

            x = fig.colorbar(surf, shrink=0.5, aspect=5)
            x.set_label('%s' % counter + 'polonies')
            counter+=1


        plt.savefig('tutte_single_point.png')
        plt.show()

        plt.close()
        counter=0
        for i in self.spring_kernellist:

            z = np.reshape(i(plotset).T, xx.shape)
            z += counter

            surf = ax.plot_surface(xx, yy, z, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')

            x = fig.colorbar(surf, shrink=0.5, aspect=5)
            x.set_label('%s' % counter + 'polonies')
            counter+=1

        plt.savefig('spring_single_point.png')
        plt.show()

        plt.close()


    def plot_centroid_error_scatter(self ):


        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        for i in range(len(self.spring_centroid_corerror)):
            color = [random.random(), random.random(), random.random()]

            for spot in self.tutte_centroid_corerror[i]:
                ax.scatter(spot[0],
                           spot[1], self.polony_counts[i],
                           color=color)
        plt.title('Tutte reconstruction')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_zlabel('polony number')
        ax = fig.add_subplot(1, 2, 2, projection='3d')

        for i in range(len(self.spring_centroid_corerror)):
            color = [random.random(), random.random(), random.random()]

            for spot in self.spring_centroid_corerror[i]:
                ax.scatter(spot[0],
                           spot[1], self.polony_counts[i],
                           color=color)
        plt.title('Spring reoconstruction')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_zlabel('polony number')
        plt.savefig('centroid_grid_size_error_new.png')
        plt.close()

    def his2d_experiment(self):
        tittle = 'hist2d'
        master_directory = parsift_lib.prefix(tittle)
        counter=0
        for i in self.tutte_centroid_corerror:
            i=np.array(i)
            plt.hist2d(i[:,0],i[:,1])
            counter+=1

            plt.savefig(master_directory + '/' + '%s' % counter + 'tutte.png')
            plt.close()
        counter=0
        for i in self.spring_centroid_corerror:
            i=np.array(i)
            plt.hist2d(i[:,0] ,
                       i[:,1])
            counter+=1
            plt.savefig(master_directory + '/' + '%s' % counter + 'spring.png')
            plt.close()

    def his1d_experiment(self):

        tittle = 'hist1d'
        master_directory = parsift_lib.prefix(tittle)



        for i in range(len(self.tutte_centroid_distance_error)):
            counter = self.polony_counts[i * self.repeat]
            sns.distplot(self.tutte_centroid_distance_error[i], norm_hist=True, kde=True,
                         kde_kws={'linewidth': 3},
                         label=('ponoly number'+'%s'%counter))
        plt.legend(prop={'size': 16}, title = 'Polony number')

        plt.title('density plot of distance error from tutte reconstruction')
        plt.savefig(master_directory + '/' + '%s' % i + 'tutte_distance_error.png')

        plt.close()
        for i in range(len(self.spring_centroid_distance_error)):
            counter = self.polony_counts[i * self.repeat]
            sns.distplot(self.spring_centroid_distance_error[i], norm_hist=True, kde=True,
                         kde_kws={'linewidth': 3},
                         label=('ponoly number' + '%s' % counter))
        plt.legend(prop={'size': 16}, title='Polony number')

        plt.title('density plot of distance error from spring reconstruction')
        plt.savefig(master_directory + '/' + '%s' % i + 'spring_distance_error.png')

        plt.close()




    def his1d_experiment_repeat(self):

        title = 'hist1d_repeat'
        master_directory = parsift_lib.prefix(title)
        tutte_merge_list=[]
        spring_merge_list=[]
        for i in range(int(len(self.seedlist)/self.repeat)):
            tutte_merge_list.append(np.concatenate(self.tutte_centroid_distance_error[(i*self.repeat):(i+1)*self.repeat],axis=None))
        for i in range(int(len(self.seedlist)/self.repeat)):
            spring_merge_list.append(np.concatenate(self.spring_centroid_distance_error[(i*self.repeat):(i+1)*self.repeat],axis=None))


        for i in range(len(tutte_merge_list)):
            counter=self.polony_counts[i*self.repeat]

            n,bins,patches=plt.hist(tutte_merge_list[i], alpha=0.5, label=('%s' % i),normed=True)
            mu,sigma=norm.fit(tutte_merge_list[i])
            y=mlab.normpdf(bins,mu,sigma)
            l=plt.plot(bins,y,'r--',linewidth=2)
            plt.title('point distribution of distance error from tutte reconstruction of '+'%s'%counter+'polonies'+'\n fit by gaussian distribution')
            plt.legend()
            plt.savefig(master_directory + '/' + '%s' % i + 'tutte.png')

            plt.close()
        for i in range(len(tutte_merge_list)):
            counter = self.polony_counts[i * self.repeat]
            sns.distplot(tutte_merge_list[i], norm_hist=True, kde=True,
                         kde_kws={'linewidth': 3},
                         label=('ponoly number'+'%s'%counter))

            plt.title(
                'distribution of distance error from tutte reconstruction of ' + '%s' % counter + 'polonies'+'\n fit by kernel density estimation')
            plt.savefig(master_directory + '/' + '%s' % i + 'kernel_tutte.png')

            plt.close()
        for i in range(len(spring_merge_list)):
            n,bins,pathces=plt.hist(spring_merge_list[i], alpha=0.5, label='%s' % i,normed=True)
            mu, sigma = norm.fit(spring_merge_list[i])
            y = mlab.normpdf(bins, mu, sigma)
            l = plt.plot(bins, y, 'r--', linewidth=2)
            plt.title('point distribution of distance error from spring reconstruction of '+'%s'%counter+'polonies'+'\n fit by guassian distribition')
            plt.savefig(master_directory + '/' + '%s' % i + 'spring.png')
            plt.close()
        for i in range(len(spring_merge_list)):
            counter = self.polony_counts[i * self.repeat]
            sns.distplot(spring_merge_list[i], norm_hist=True, kde=True,
                         kde_kws={'linewidth': 3},
                         label=('ponoly number'+'%s'%counter))

            plt.title(
                'distribution of distance error from spring reconstruction of ' + '%s' % counter + 'polonies'+'\n fit by kernel density estimation')
            plt.savefig(master_directory + '/' + '%s' % i + 'kernel_spring.png')

            plt.close()


def normal_coord(exp,normallist):
    unit_seedlist=[]
    for i in range(len(exp.seedlist)):
        unit_seedlist.append(exp.seedlist[i]/normallist[i])
    unit_sitelist=[]
    for i in range(len(exp.sitelist)):
        unit_sitelist.append(exp.sitelist[i]/normallist[i])

    return unit_seedlist,unit_sitelist
def kernel_per_experiment(springerror,tutteerror):
    springkernels=[]
    tuttekernels=[]
    for i in range(len(springerror)):
        springkernels.append(st.gaussian_kde(np.stack([springerror[i][:,0],springerror[i][:,1]])))
    for i in range(len(tutteerror)):
        tuttekernels.append(st.gaussian_kde(np.stack([tutteerror[i][:,0],tutteerror[i][:,1]])))
    return springkernels,tuttekernels
def normal_error_list(exp,unitseedlist):
    errorlist_spring = []

    for i in range(len(exp.spring_adjusted_seed)):
        errorlist_spring.append(exp.spring_adjusted_seed[i] - unitseedlist[i])
    errorlist_tutte = []
    for i in range(len(exp.tutte_adjusted_seed)):
        errorlist_tutte.append(exp.tutte_adjusted_seed[i] - unitseedlist[i])
    return errorlist_spring,errorlist_tutte

def test_kernel_list_error_fit_plot(seedlist,kernelist,polonynum):
    xx, yy = np.mgrid[(- 2):(+ 2):100j, ( - 2):(+ 2):100j]
    plotset = np.vstack([xx.ravel(), yy.ravel()])
    fig = plt.figure(figsize=(13, 7))

    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.set_zlabel('PDF')
    ax.set_title('Tutte reconstruction Point distribution')
    for i in range(len(kernelist)):





        z=np.reshape(kernelist[i](plotset).T,xx.shape)
        z+=i

        surf = ax.plot_surface(xx, yy, z, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')


        x=fig.colorbar(surf, shrink=0.5, aspect=5)
        x.set_label('%s'%polonynum[i]+'polonies')

    plt.savefig('all_pdf_test_2.png')
    plt.show()

    plt.close()
def get_pol_centroid(exp):
    polcentroidlist=[]
    for i in range(len(exp.seedlist)):
        polcentroidlist.append(get_point_polony(exp.seedlist[i],exp.seedlist[i]))
    exp.polcentroidlist=polcentroidlist
def list_centroid_error(exp):
    tutte_centroidlist=[]
    tutte_error=[]
    spring_error=[]
    spring_centroidlist=[]
    for i in range(len(exp.seedlist)):
        track_centroid=track_spot_centroid(exp.polcentroidlist[i],exp.seedlist[i],exp.spring_adjusted_seed[i],exp.tutte_adjusted_seed[i])
        temp_error_tutte,tutte_temporarylist=track_centroid.tutte_centroid_error()
        temp_error_spring,spring_temporarylist=track_centroid.spr_centroid_error()
        tutte_centroidlist.append(tutte_temporarylist)
        tutte_error.append(temp_error_tutte)
        spring_error.append(temp_error_spring)
        spring_centroidlist.append(spring_temporarylist)
    exp.spring_centroidlist=spring_centroidlist
    exp.spring_centroid_error=spring_error
    exp.tutte_centroid_error=tutte_error
    exp.tutte_centroidlist=tutte_centroidlist
def normal_centroid_error_list(exp):
    errorlist_spring = []

    for i in range(len(exp.spring_centroidlist)):
        errorlist_spring.append(exp.spring_centroidlist[i] - exp.seedlist[i])
    errorlist_tutte = []
    for i in range(len(exp.tutte_adjusted_seed)):
        errorlist_tutte.append(exp.tutte_centroidlist[i] - exp.seedlist[i])
    return errorlist_spring,errorlist_tutte

def plot_centroid_error_scatter(exp,unitseedlist):

    errormat_spring=[]

    for i in range(len(exp.spring_centroidlist)):
        errormat_spring.append(exp.spring_centroidlist[i]-unitseedlist[i])
    errormat_tutte = []
    for i in range(len(exp.tutte_centroidlist)):
        errormat_tutte.append(exp.tutte_centroidlist[i] - unitseedlist[i])

    fig=plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    for i in range(len(errormat_spring)):
        color = [random.random(), random.random(), random.random()]

        for spot in errormat_spring[i]:

            ax.scatter(spot[0],
                   spot[1], exp.polony_counts[i],
                   color=color)
    plt.title('Tutte reconstruction')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.set_zlabel('polony number')
    ax = fig.add_subplot(1, 2, 2, projection='3d')

    for i in range(len(errormat_tutte)):
        color = [random.random(), random.random(), random.random()]

        for spot in errormat_tutte[i]:
            ax.scatter(spot[0],
                       spot[1], exp.polony_counts[i],
                       color=color)
    plt.title('Spring reoconstruction')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.set_zlabel('polony number')
    plt.savefig('centroid_grid_size_error_new.png')
    plt.close()


def his2d_experiment(exp):
    tittle='hist2d'
    master_directory=parsift_lib.prefix(tittle)

    for i in range(len(exp.tutte_centroidlist)):
        plt.hist2d(np.array(exp.tutte_centroidlist[i])[:,0]-np.array(exp.seedlist[i])[:,0],np.array(exp.tutte_centroidlist[i])[:,1]-np.array(exp.seedlist[i])[:,1])

        plt.savefig(master_directory+'/'+'%s'%i+'tutte.png')
    for i in range(len(exp.spring_centroidlist)):
        plt.hist2d(np.array(exp.spring_centroidlist[i])[:,0]-np.array(exp.seedlist[i])[:,0],np.array(exp.spring_centroidlist[i])[:,1]-np.array(exp.seedlist[i])[:,1])

        plt.savefig(master_directory+'/'+'%s'%i+'spirng.png')

def his1d_experiment(exp):

    tittle='hist1d'
    master_directory=parsift_lib.prefix(tittle)

    for i in range(len(exp.tutte_centroid_error)):
        plt.hist(exp.tutte_centroid_error[i],alpha=0.5,label='%s'%i)

    plt.savefig(master_directory+'/'+'%s'%i+'tutte.png')
    plt.close()
    for i in range(len(exp.spring_centroid_error)):
        plt.hist(exp.spring_centroid_error[i],alpha=0.5,label='%s'%i)

    plt.savefig(master_directory+'/'+'%s'%i+'spring.png')

    plt.close()


def his1d_experiment(f):

        tittle = 'hist1d'
        master_directory = parsift_lib.prefix(tittle)



        for i in range(len(f.tutte_centroid_distance_error)):
            counter = f.polony_counts[i * f.repeat]
            sns.distplot(f.tutte_centroid_distance_error[i], hist=False, kde=True,
                         kde_kws={'linewidth': 3},
                         label=('ponoly number'+'%s'%counter))
        plt.legend(prop={'size': 16}, title = 'Polony number')
        plt.xlabel('Distance')
        plt.ylabel('Density')

        plt.title('density plot of distance error from tutte reconstruction')
        plt.savefig(master_directory + '/' + '%s' % i + 'tutte_distance_error.png')

        plt.close()
        for i in range(len(f.spring_centroid_distance_error)):
            counter = f.polony_counts[i * f.repeat]
            sns.distplot(f.spring_centroid_distance_error[i], hist=False, kde=True,
                         kde_kws={'linewidth': 3},
                         label=('ponoly number' + '%s' % counter))
        plt.legend(prop={'size': 16}, title='Polony number')
        plt.xlabel('Distance')
        plt.ylabel('Density')

        plt.title('density plot of distance error from spring reconstruction')
        plt.savefig(master_directory + '/' + '%s' % i + 'spring_distance_error.png')

        plt.close()

def kernel_all_seed(analyze,n):
    kernelspring=analyze.spring_kernellist[n]
    kernel_tutte=analyze.tutte_kernellist[n]
    seedori=analyze.polcentroidlist[n]
    seedspring=analyze.spring_centroidlist[n]
    seedtutte=analyze.tutte_centroidlist[n]
    for i in seedori:
        x,y=i[0]



def plot_2d_kernel_densityplot(ana):
    counter=0
    for i in ana.spring_centroid_corerror:
        df=pd.DataFrame(i,columns=['x','y'])
        sns.jointplot(x='x',y='y',data=df,kind='kde')
        num=int(ana.polony_counts[counter])
        plt.title('%s'%num+'polonies')
        plt.savefig('spring_relaxation'+'%s'%counter+'ponolies.png')
        counter+=1
        plt.close()
    counter = 0
    for i in ana.tutte_centroid_corerror:
        df = pd.DataFrame(i, columns=['x', 'y'])
        sns.jointplot(x='x', y='y', data=df, kind='kde')
        num = int(ana.polony_counts[counter])
        plt.title('%s' % num + 'polonies')
        plt.savefig('tutte_embedding' + '%s' % counter + 'ponolies.png')
        counter += 1
        plt.close()


def plot_2d_kernel_by_point(exp,nl):
    tutte,spring,original=extract_adjusted_px_polony(exp,nl)
    tutte=np.array(tutte)
    spring=np.array(spring)
    tutte_by_point=np.zeros((4*nl+1,len(tutte),2))
    spring_by_point=np.zeros((4*nl+1,len(spring),2))

    for i in range(4*nl+1):
        tutte_by_point[i]=tutte[:,i,:]
        spring_by_point[i]=spring[:,i,:]

    for i in range(4*nl+1):

        tutte_df=pd.DataFrame(tutte_by_point[i],columns=['x','y'])
        sns.jointplot(x='x',y='y',data=tutte_df,kind='kde')
        plt.savefig('tutte'+'%s'%i+'.png')
        plt.close()

    for i in range(4*nl+1):

        tutte_df=pd.DataFrame(spring_by_point[i],columns=['x','y'])
        sns.jointplot(x='x',y='y',data=tutte_df,kind='kde')
        plt.savefig('spring'+'%s'%i+'.png')
        plt.close()


def plot_2d_kernel_by_point_unadjusted(exp,nl):
    tutte,spring=extract_px_polony(exp,nl)
    tutte=np.array(tutte)
    spring=np.array(spring)
    tutte_by_point=np.zeros((4*nl+1,len(tutte),2))
    spring_by_point=np.zeros((4*nl+1,len(spring),2))

    for i in range(4*nl+1):
        tutte_by_point[i]=tutte[:,i,:]
        spring_by_point[i]=spring[:,i,:]

    for i in range(4*nl+1):

        tutte_df=pd.DataFrame(tutte_by_point[i],columns=['x','y'])
        sns.jointplot(x='x',y='y',data=tutte_df,kind='kde')
        plt.savefig('unadjusted_tutte'+'%s'%i+'.png')
        plt.close()

    for i in range(4*nl+1):

        tutte_df=pd.DataFrame(spring_by_point[i],columns=['x','y'])
        sns.jointplot(x='x',y='y',data=tutte_df,kind='kde')
        plt.savefig('unadjusted_spring'+'%s'%i+'.png')
        plt.close()


def repeats_by_point(exp,nl):
    tutte,spring,original=extract_adjusted_px_polony(exp,nl)
    tutte=np.array(tutte)
    spring=np.array(spring)
    tutte_by_point=np.zeros((4*nl+1,len(tutte),2))
    spring_by_point=np.zeros((4*nl+1,len(spring),2))

    for i in range(4*nl+1):
        tutte_by_point[i]=tutte[:,i,:]
        spring_by_point[i]=spring[:,i,:]
    return tutte_by_point,spring_by_point,original

def extract2points(exp,points):
    grid=np.array(exp.sitelist[0])
    scale=max(grid[:,0])
    points=scale*np.array(points)
    tuttelist=list()
    springlist=list()
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    original=list()
    for i in range(len(exp.seedlist)):
        id=get_point_polony(points,exp.seedlist[i])
        original.append(points)
        print id
        estimation=track_spot_centroid(id,exp.seedlist[i],exp.springallpolony,exp.tutteallpolony)
        tutte_error,tutte_spot=estimation.tutte_centroid_error()
        print tutte_spot
        color = [random.random(), random.random(), random.random()]
        for spot in tutte_spot:
            ax.scatter (spot[0],
                    spot[1], exp.polony_counts[i],
                   color=color)

    plt.title('Tutte reconstruction')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('polony number')
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    for i in range(len(exp.seedlist)):
        id = get_point_polony(points, exp.seedlist[i])
        print id
        estimation = track_spot_centroid(id, exp.seedlist[i], exp.springallpolony[i], exp.tutteallpolony[i])
        spring_error, spring_spot = estimation.spr_centroid_error()
        color = [random.random(), random.random(), random.random()]
        for spot in spring_spot:

            ax.scatter(spot[0],
                   spot[1], exp.polony_counts[i],
                   color=color)
        tuttelist.append(tutte_spot)
        springlist.append(spring_spot)
    plt.title('Spring reoconstruction')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('polony number')

    plt.show()

    fig.savefig('cross.png')
    plt.close()

    return tuttelist,springlist

def repeat2points(exp,points):
    tutte,spring=extract2points(exp,points)
    tutte=np.array(tutte)
    spring=np.array(spring)
    tutte_by_point=np.zeros((2,len(tutte),2))
    spring_by_point=np.zeros((2,len(spring),2))

    for i in range(2):
        tutte_by_point[i]=tutte[:,i,:]
        spring_by_point[i]=spring[:,i,:]
    return tutte_by_point,spring_by_point
def draw_recon(exp):
    pollist = np.zeros((len(exp.seedlist)))
    for i in range(len(exp.seedlist)):
        a = get_point_polony([[7, 0]], exp.seedlist[i])

        pollist[i] = int(a[0][1])

    for i in range(len(exp.springallpolony)):
        plt.scatter(exp.springallpolony[i][int(pollist[i]), 0], exp.springallpolony[i][int(pollist[i]), 1])


def coarse_adjustment(exp):
    shift_vector=np.zeros((len(exp.seedlist),2))
    scale=max(exp.sitelist[0][:,0])
    for i in range(len(exp.seedlist)):
        error=exp.springallpolony[i]-np.array(exp.seedlist[i])/scale
        shift_vector[i]=[np.average(error[:,0]),np.average(error[:,1])]
    print shift_vector
    coarse_adjusted=[]
    for i in range(len(exp.seedlist)):
        coarse_adjusted.append(np.array(exp.springallpolony[i])-np.array(shift_vector[i]))
    exp.spring_coarseadjsuted=coarse_adjusted
def draw_recon2points(exp,points):
    x = []
    y = []
    points_repeat=[]
    for j in points:
        xx=[]
        yy=[]
        pollist = np.zeros((len(exp.springallpolony)))

        for i in range(len(exp.seedlist)):
            if len(exp.seedlist[i])>1:
                a = get_point_polony([j], exp.seedlist[i])
                pollist[i] = int(a[0][1])
        print pollist
        for i in range(len(exp.seedlist)):

            if len(exp.seedlist[i])>1:
                xx.append(exp.springallpolony[i][int(pollist[i]),0])
                yy.append(exp.springallpolony[i][int(pollist[i]),1])
                x.append(exp.springallpolony[i][int(pollist[i]),0])
                y.append(exp.springallpolony[i][int(pollist[i]),1])
        points_repeat.append([[xx[q],yy[q] ]for q in range(len(xx))])
    plt.hist2d(x,y)
    return points_repeat



def align_two_vector(v_ref,v_input):
    x0=v_ref[0]
    y0=v_ref[1]
    x1=v_input[0]
    y1=v_input[1]
    cosine=(x0*x1+y1*y0)/(math.sqrt(x0**2+y0**2)*math.sqrt(x1**2+y1**2))
    print v_ref,v_input,(x0*x1+y1*y0)/(math.sqrt(x0**2+y0**2)*math.sqrt(x1**2+y1**2))
    if cosine>1:
        cosine=1
    theta=math.acos(cosine)

    cross=x1*y0-y1*x0
    if cross>0:
        rot_x1=math.cos(theta)*x1-math.sin(theta)*y1
        rot_y1=math.sin(theta)*x1+math.cos(theta)*y1
    else:
        rot_x1=math.cos(theta)*x1+math.sin(theta)*y1
        rot_y1=-math.sin(theta)*x1+math.cos(theta)*y1
    #plt.plot([0,x0],[0,y0],color='red',linewidth=2)
    #plt.plot([0,x1],[0,y1],color='green')
    #plt.plot([0,rot_x1],[0,rot_y1],color='blue')
    #plt.show()
    return [rot_x1,rot_y1]

def multiple_align(points_repeat):
    points_vector=np.array(points_repeat[1])-np.array(points_repeat[0])
    rot_points=np.zeros((len(points_repeat[0]),2))
    for i in range(len(points_vector)):
        print i
        rot_points[i]=align_two_vector(points_vector[0],points_vector[i])
    shift_rot_points=rot_points+np.array(points_repeat[0])
    rot_two=[shift_rot_points,points_repeat[0]]
    all=np.vstack([shift_rot_points,np.array(points_repeat[0])])
    plt.hist2d(all[:,0],all[:,1],cmap='viridis')
    return rot_two,all

def draw_FWHM(dt,rmin,rmax):
    dt=np.array(dt)
    sns.distplot(dt,hist=False,kde_kws={'shade':True})
    dt=np.sort(dt)
    kernel=st.gaussian_kde(dt)
    kyy=kernel(dt)
    bin=len(dt)/2
    p1=argrelmax(kyy[0:bin])[0][0]
    p2=bin+argrelmax(kyy[bin:len(dt)])[0][0]
    hf1=kyy[p1]/2
    hf2=kyy[p2]/2
    sep=p1+argrelmin(kyy[p1:p2])[0][0]
    vhf1=find_half_kernel(hf1,kernel,rmin,dt[p1],steps=1000)
    vhf2=find_half_kernel(hf1,kernel,dt[p1],dt[sep],steps=1000)
    vhf3=find_half_kernel(hf2,kernel,dt[sep],dt[p2],steps=1000)
    vhf4=find_half_kernel(hf2,kernel,dt[p2],rmax,steps=1000)
    fwhm1=vhf2-vhf1
    fwhm2=vhf4-vhf3
    use_this_color=sns.color_palette('deep')[0]

    sns.distplot(dt,kde_kws={'shade':True},hist=False,color=use_this_color)
    plt.vlines(x=vhf1,ymin=0,ymax=1.3,linestyles=':',color=use_this_color)
    plt.vlines(x=vhf2,ymin=0,ymax=1.3,linestyles=':',color=use_this_color)
    plt.vlines(x=vhf3,ymin=0,ymax=1.3,linestyles=':',color=use_this_color)
    plt.vlines(x=vhf4,ymin=0,ymax=1.3,linestyles=':',color=use_this_color)
    plt.text(x=vhf1,y=1.2,s='%s'%fwhm1)
    plt.text(x=vhf3,y=1.2,s='%s'%fwhm2)
    plt.savefig('fwhm.png')






def find_half_kernel(halfvalue,kernel,xmin,xmax,steps=100):
    allxs=[xmin+i*(xmax-xmin)/steps for i in range(steps)]
    allys=kernel(allxs)
    hf=np.argmin(abs(allys-halfvalue))
    hfx=allxs[hf]
    return hfx
