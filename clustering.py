# Reading trajectories
import MDAnalysis as mda
from MDAnalysis.analysis import rdf

# Python tools
import matplotlib.pyplot as plt
import numpy as np
import pickle
import datetime
import sys
import os

# Optimize python codes via numba
from numba import jit

# Linkage modules
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


# MDAnalysis code for reading LJ systems
# MDAnalysis currently do not support "LJ" units of LAMMPS
# So, I assign the units : 
# - ps=tau 
# - sigma=A

# MDAnalysis implented method for calculating RDF
#g1 = u.select_atoms('index 0:49999')
#rdf = InterRDF(g1, g1)
#rdf.run()
#plt.plot(rdf.bins, rdf.rdf)

# ------------------------------#
# Get coordintaes from universe #
# ------------------------------#

# 1) from atom types
# example: ith atom jth trajectory
#     coordinates = u.trajectory.timeseries()
#     coordinates[i][j] = [ x y z ]

# 2) from trajectory
# example : ith atom jth trajectory
#     coordinates = u.trajectory[j]
#     coordinates[i] = [ x y z ]

# -------------------------------#
# Get residue info from universe #
# -------------------------------#
# r1 = residues[0] # First chain
# r1.atoms.ids : Atom indexs
# r1.atoms.types : Atoms types

@jit(nopython=True)
def ovalue(v1,v2):
    dist1 = ( (v1[0] - v2[0])**2 +
              (v1[1] - v2[1])**2 +
              (v1[2] - v2[2])**2 )**0.5

    dist2 = ( (v1[0] + v2[0])**2 +
              (v1[1] + v2[1])**2 +
              (v1[2] + v2[2])**2 )**0.5

    return min(dist1,dist2)

@jit(nopython=True)
def omatrix(vector, ijmat, out):
    id = 0
    for ipart in range(len(vector)-1):
        for jpart in range(ipart+1,len(vector)):

            if(ijmat[ipart][jpart] == 0):
                out[id] = 10
                id+=1
                continue

            #oval = ovalue(vector[ipart],vector[jpart])
            #distance_matrix = np.append(distance_matrix, oval)
            #distance_matrix[id] = oval
            #id += 1

            v1 = vector[ipart]
            v2 = vector[jpart]
            dist1 = ( (v1[0] - v2[0])**2 +
            (v1[1] - v2[1])**2 +
            (v1[2] - v2[2])**2 )**0.5

            dist2 = ( (v1[0] + v2[0])**2 +
            (v1[1] + v2[1])**2 +
            (v1[2] + v2[2])**2 )**0.5
            oval = min(dist1, dist2)
            if (id%100000000==0): print(id)
            out[id] = oval
            id += 1

    #if (len(vector)*(len(vector)-1)/2) == len(distance_matrix) :
    #    print("Distance matrix set-up : SUCESS") 
    return out

@jit(nopython=True)
def nearest_check(coordinates, boxes, itraj, ijmat, rcut=2.5):
    natoms = len(coordinates)
    for ipart in range(natoms-1):
        for jpart in range(ipart+1,natoms):
        # Initialize comb matrix
            ijmat[ipart][jpart] = 0
            ijmat[jpart][ipart] = 0

            dx=coordinates[ipart][itraj][0]-coordinates[jpart][itraj][0]
            dy=coordinates[ipart][itraj][1]-coordinates[jpart][itraj][1]
            dz=coordinates[ipart][itraj][2]-coordinates[jpart][itraj][2]

            # perodic boundary conditions for orthogonal cells
            # code must be modified for triclinic cells
            dx -= boxes[itraj][0] * round(dx/boxes[itraj][0])
            dy -= boxes[itraj][1] * round(dy/boxes[itraj][1])
            dz -= boxes[itraj][2] * round(dz/boxes[itraj][2])
            vv = dx*dx + dy*dy + dz*dz
            
            if (vv <= rcut**2): 
                ijmat[ipart][jpart] = 1
                ijmat[jpart][ipart] = 1

    return ijmat	

class traj_anlysis:

    def __init__(self, trajfile, initfile, integ_ts=0.005, freq=20000):
        self.trajfile = trajfile
        self.initfile = initfile
        self.integ_ts = integ_ts
        self.freq = freq

        self.u =mda.Universe(self.initfile, # LAMMPS initial file *.data
                             self.trajfile,      # Trajectory file *.dcd
                             atom_style='id resid type x y z',
                             dt=self.integ_ts*self.freq)

        self.natoms=self.u.atoms.n_atoms
        self.nmoles=self.u.atoms.n_residues
        self.types =self.u.residues.types
        #self.num_moles = [ u.residues[i].atoms.n_atoms for i in range(self.nmoles) ]

        self.nframes = self.u.trajectory.n_frames
        self.coordinates = self.u.trajectory.timeseries()
        self.cluster_ids = []

        self.boxes = np.empty((0,6), float)
        for itraj in range(self.nframes):
            self.boxes = np.append(self.boxes, [self.u.trajectory[itraj].dimensions], axis = 0)

    def vector_assign(self, itraj):
        prev = 0
        vector = []
        for iresidues in range(self.nmoles):
            # First atoms at each residues
            point=prev
            dx=self.coordinates[point+1][itraj][0] - self.coordinates[point][itraj][0]
            dy=self.coordinates[point+1][itraj][1] - self.coordinates[point][itraj][1]
            dz=self.coordinates[point+1][itraj][2] - self.coordinates[point][itraj][2]
            sqvv=(dx**2+dy**2+dz**2)**0.5
            vector.append([dx/sqvv, dy/sqvv, dz/sqvv])

            # Middle atoms
            for iparticles in range(1,len(self.types[iresidues])-1):
                point=prev+iparticles
                dx=self.coordinates[point+1][itraj][0] - self.coordinates[point-1][itraj][0]
                dy=self.coordinates[point+1][itraj][1] - self.coordinates[point-1][itraj][1]
                dz=self.coordinates[point+1][itraj][2] - self.coordinates[point-1][itraj][2]
                sqvv=(dx**2+dy**2+dz**2)**0.5
                vector.append([dx/sqvv, dy/sqvv, dz/sqvv])

            # Last atoms at each residues
            point=prev+len(self.types[iresidues])-1
            dx=self.coordinates[point][itraj][0] - self.coordinates[point-1][itraj][0]
            dy=self.coordinates[point][itraj][1] - self.coordinates[point-1][itraj][1]
            dz=self.coordinates[point][itraj][2] - self.coordinates[point-1][itraj][2]
            sqvv=(dx**2+dy**2+dz**2)**0.5
            vector.append([dx/sqvv, dy/sqvv, dz/sqvv])
            prev += len(self.types[iresidues])

        return vector

    def print_xyz(self, itraj, filename="trajectory.xyz", mode='w'):
        cids = self.cluster_ids[itraj]
        size = np.unique(cids, return_counts=True)
        size_dict = dict()
        for (id, num) in zip(size[0],size[1]):
            size_dict[id] = num

        with open(filename, mode) as fw:
            fw.write(f'{self.natoms}\n')
            fw.write(f'Lattice="{ta.boxes[itraj][0]} 0.0 0.0 0.0 {ta.boxes[itraj][1]} 0.0 0.0 0.0 {ta.boxes[itraj][2]}" \n')

            for id in range(self.natoms):
                xyz = self.coordinates[id][itraj]
                cid = cids[id]
                mid = self.u.atoms[id].resid
                fw.write(f'C {xyz[0]} {xyz[1]} {xyz[2]} {cid} {size_dict[cid]} {mid}\n')

    def load_cids(self, filename="cluster.pickle"):
        with open(filename,"rb") as fr:
            self.cluster_ids = pickle.load(fr)


if __name__ ==  '__main__':
    print("Program start:",datetime.datetime.now())
    if len(sys.argv) == 1 :
        print("Run 'python clustering.py --help'")
        sys.exit()

    if ( str(sys.argv[1]) == '--help' or str(sys.argv[1]) == '-h' ) :
        print("usage : python clustering.py [traj filename] [lammps initfile] [args] \n",
              "-c : use previously obtained cluster_ids \n",
              "-p : print ovito read-able trajectory file \n")
        sys.exit()

    if len(sys.argv) < 3 :
        print("Run 'python clustering.py --help'")
        sys.exit()

    flags = [ flag for flag in sys.argv if "-" in flag ]
    trajfile = str(sys.argv[1])
    initfile = str(sys.argv[2])
    ta = traj_anlysis(trajfile, initfile)

    # Check print files
    if ( "-p" in sys.argv ):
        print_filename = str(sys.argv[sys.argv.index("-p")+1])
        if ( os.path.isfile(print_filename) ):
            print(f'[{print_filename}] already exists. Please check the file')
            sys.exit()
    
    # Check cluster files
    if ( "-c" in sys.argv ):
        cluster_filename = str(sys.argv[sys.argv.index("-c") +1])
        ta.load_cids(cluster_filename)
    
    if ( "-c" not in sys.argv ):
        with open("fraction.dat", "w") as fw:
            fw.write("# time frac num_clusters\n")
        
        for i in range(ta.nframes):
            print(i,"th trajectory...")
            comb = np.zeros([ta.natoms,ta.natoms])
            comb = nearest_check(ta.coordinates, ta.boxes, i, comb)
            vector = ta.vector_assign(i) # Last trajectory
            vector = np.array(vector, dtype=np.float32)
            out = np.zeros([int(len(vector)*(len(vector)-1)/2)], dtype=np.float32)
            print("Initialize + Assign vectors:",datetime.datetime.now())
            dist = omatrix(vector, comb, out)
            del comb
            del out
            del vector
            print("Distance matrix :",datetime.datetime.now())
            
            #plt.hist(dist, bins=40)
            #plt.show()
            
            # Clustering
            Z=linkage(dist,'single')
            print("SLINK:",datetime.datetime.now())
           
            # Flatten clusters
            ta.cluster_ids.append( fcluster(Z, t=0.1, criterion='distance') )
            print("Flatten clusters:",datetime.datetime.now())
            
            #fig = plt.figure(figsize=(25, 10))
            #dn = dendrogram(Z)
            #plt.show()
        
            sum  = 0
            ncluster = 0
            keep = np.unique(np.unique(ta.cluster_ids[i], return_counts=True)[1], return_counts=True)
            for nmon, nclus in zip(keep[0],keep[1]):
                if(nmon >= 50):
                    sum += nmon*nclus
                    ncluster += 1
            with open("fraction.dat", "a") as fa:
                fa.write(f"{ta.u.trajectory[i].time} {sum/ta.natoms} {ncluster}\n")
        
        with open("cluster.pickle","wb") as fw:
            pickle.dump(ta.cluster_ids,fw)
   
    if ( "-p" in sys.argv ):
        print_filename = str(sys.argv[sys.argv.index("-p")+1])
        for i in range(ta.nframes):
            ta.print_xyz(itraj=i, filename=print_filename, mode='a')

