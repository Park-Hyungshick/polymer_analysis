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
        self.morph_ids = []
        self.vectors = []

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

    def print_xyz(self, itraj, filename="trajectory.xyz", mode='w', anlysis=False):
        cids = self.cluster_ids[itraj]
        size = np.unique(cids, return_counts=True)
        size_dict = dict()
        for (id, num) in zip(size[0],size[1]):
            size_dict[id] = num

        with open(filename, mode) as fw:
            fw.write(f'{self.natoms}\n')
            fw.write(f'Lattice="{self.boxes[itraj][0]} 0.0 0.0 0.0 {self.boxes[itraj][1]} 0.0 0.0 0.0 {self.boxes[itraj][2]}" \n')

            for id in range(self.natoms):
                xyz = self.coordinates[id][itraj]
                cid = cids[id]
                mid = self.u.atoms[id].resid
                if ( anlysis ):
                    fw.write(f'C {xyz[0]} {xyz[1]} {xyz[2]} {cid} {size_dict[cid]} {mid} {self.morph_ids[itraj][id]} \n')
                else:
                    fw.write(f'C {xyz[0]} {xyz[1]} {xyz[2]} {cid} {size_dict[cid]} {mid}\n')

    def load_cids(self, filename="cluster.pickle"):
        with open(filename,"rb") as fr:
            self.cluster_ids = pickle.load(fr)

    def cluster_idsize_dict(self, itraj):
        # type(cluster_dict[itraj]) will be dictionary
        cids = self.cluster_ids[itraj]
        size = np.unique(cids, return_counts=True)
        size_dict = dict()

        for (id, num) in zip(size[0],size[1]):
            size_dict[id] = num
        return size_dict

    def assign_types(self, itraj):
        # Code for assign morphology
        # First, cluster_ids needed
        # morph_types
        # Each segments are considered
        #  A - amorphous / Ci - crystalline segments included to a ith cluster
        #  
        # 'A' (A--------A) : Amorphous
        # 'D' (A-------C1) : Dangling ends
        # 'L' (C1------C1) : Loop
        # 'T' (C1------C2) : Tie
        # cid              : Crystalline monomers (Cluster size >=50)
        # Only [cid] values are int

        size_dict = self.cluster_idsize_dict(itraj)
        prev = 0
        morph = []

        for iresidues in range(self.nmoles):

            morph_chain = ['/']
            for iparticles in range(len(self.types[iresidues])):
                itype = 'A'
                id = prev+iparticles
                cid = self.cluster_ids[itraj][id]
                size = size_dict[cid]

                if size >= 50 :itype = cid

                if iparticles != 0 and prev_type != itype :
                    morph_chain.append("/")

                morph_chain.append(itype)
                prev_type = itype
            morph_chain.append('/')

            morph_chain_assign = []
            slice_idx = [ idx for idx, val in enumerate(morph_chain) if val == '/' ]
            for key in range(len(slice_idx)-1):
                segments = morph_chain[slice_idx[key]+1:slice_idx[key+1]]

                # Crystalline domain
                if type(segments[0]) is int or type(segments[0]) is np.int32:
                    morph_chain_assign.extend(segments)

                # Amorphous case
                elif len(segments) == len(self.types[iresidues]): 
                    morph_chain_assign.extend(segments)

                # Dangling ends
                elif key == 0 or key == len(slice_idx)-2 :
                    segments = ['D']*len(segments)
                    morph_chain_assign.extend(segments)

                # Tie chain
                elif morph_chain[slice_idx[key]-1] != morph_chain[slice_idx[key+1]+1] :
                    #print(morph_chain[slice_idx[key]-1], morph_chain[slice_idx[key+1]+1])
                    segments = ['T']*len(segments)
                    morph_chain_assign.extend(segments)

                # Loop chain
                elif morph_chain[slice_idx[key]-1] == morph_chain[slice_idx[key+1]+1] :
                    first = prev + len(morph_chain_assign) 
                    last  = first + len(segments) -1
                    cosine = ( self.vectors[itraj][first][0]*self.vectors[itraj][last][0] +
                               self.vectors[itraj][first][1]*self.vectors[itraj][last][1] +
                               self.vectors[itraj][first][2]*self.vectors[itraj][last][2] )

                    if cosine >= 0 :
                        segments = ['T2']*len(segments)
                    else :
                        segments = ['L']*len(segments)

                    print("Check...Loop/Tie chain")
                    print("Cosine:",cosine)
                    print("Type:",segments[0])
                    morph_chain_assign.extend(segments)
                

            prev += len(self.types[iresidues])
            morph.extend(morph_chain_assign)

        self.morph_ids.append(morph)


if __name__ ==  '__main__':
    print("Program start:",datetime.datetime.now())
    if len(sys.argv) == 1 :
        print("Run 'python clustering.py --help'")
        sys.exit()

    if ( str(sys.argv[1]) == '--help' or str(sys.argv[1]) == '-h' ) :
        print("usage : python clustering.py [traj filename] [lammps initfile] [args] \n",
              "-c : use previously obtained cluster_ids \n",
              "-p : print ovito read-able trajectory file \n",
              "-a : crystal domain analysis \n",
              "-v : change ta class variables \n",
              "-uv : change ta.u class variables \n")
        sys.exit()

    if len(sys.argv) < 3 :
        print("Run 'python clustering.py --help'")
        sys.exit()

    flags = [ flag for flag in sys.argv if "-" in flag ]
    trajfile = str(sys.argv[1])
    initfile = str(sys.argv[2])
    #if (initfile.split(".")[-1] == "dat") :
    #    print("init file format change from .dat to .data")
    #    initfile += "a"
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

        # Assign vectors
        if ( "-a" in sys.argv ):
            for i in range(ta.nframes):
                print(f"{i}th trajectory vector calculating...")
                vector = ta.vector_assign(i) 
                vector = np.array(vector, dtype=np.float32)
                ta.vectors.append(vector)

    # variable function
    for idx, value in enumerate(sys.argv):
        if ( "-v" == value ):
            variable_name = str(sys.argv[idx +1])   
            variable_value = str(sys.argv[idx +2])   
            setattr(ta,variable_name,variable_value)
        if ( "-uv" == value ):
            variable_name = str(sys.argv[idx +1])   
            variable_value = str(sys.argv[idx +2])   
            setattr(ta.u,variable_name,variable_value)
 
    if ( "-c" not in sys.argv ):
        with open("fraction.dat", "w") as fw:
            fw.write("# time frac num_clusters\n")
        
        for i in range(ta.nframes):
            print(i,"th trajectory...")
            comb = np.zeros([ta.natoms,ta.natoms])
            comb = nearest_check(ta.coordinates, ta.boxes, i, comb)
            vector = ta.vector_assign(i) 
            vector = np.array(vector, dtype=np.float32)
            out = np.zeros([int(len(vector)*(len(vector)-1)/2)], dtype=np.float32)
            print("Initialize + Assign vectors:",datetime.datetime.now())
            dist = omatrix(vector, comb, out)
            ta.vectors.append(vector)
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
   
    # Crystal morphlogy anlysis
    if ( "-a" in sys.argv ):
        for i in range(ta.nframes):
            ta.assign_types(i)

    if ( "-p" in sys.argv ):
        print_filename = str(sys.argv[sys.argv.index("-p")+1])
        for i in range(ta.nframes):
            if ("-a" in sys.argv):
                ta.print_xyz(itraj=i, filename=print_filename, 
                             mode='a',anlysis=True)
            else :
                ta.print_xyz(itraj=i, filename=print_filename, mode='a')


