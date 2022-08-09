import sys
import pickle
import numpy as np
from scipy.io import FortranFile

with open(sys.argv[1],"rb") as fr:
  clusters=pickle.load(fr)

filename=sys.argv[1].split(".")[0] + ".cids"

binfile = FortranFile(filename, "w") # write binary format file

print("Dimension : (",len(clusters),",",len(clusters[0]),")")

binfile.write_record(np.int32(len(clusters)))
binfile.write_record(np.int32(len(clusters[0])))
binfile.write_record(np.array(clusters, dtype=np.int32).T)
