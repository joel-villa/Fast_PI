"""
Goal: analyze typical spectral norm distribution of Suitesparse matrices
"""

import matplotlib.pyplot as plt

from .tests import spect_norms as sn

if __name__ == '__main__':
    mats = [
        "bcsstk08",
        "Erdos02",
        "California", 

        "494_bus", # Invalid for mag-based sparsifification
        "Harvard500", 

        # "gre_343",
        # "bcsstm07",
        # "bcsstk19", #TODO: this has odd behavior w/ proven tests (not starting at zero)
        # "hor_131",
        # "fs_541_1",
        # "bcsstk07", 
        # "bp_0",
        # "tomography", 
        # "cage7", 
        # "bcspwr06",
        # "gre_1107",
        # "can_229", 
        # "dwt_193",
        # "gre_1107",
        # "impcol_d", # NOT POSITIVE DEFINITE, doesn't work with nystrom sampling
        # "bibd_13_6", # RECTANGULAR: Doesn't work with Nystrom
    ]
    

    # Histogram
    bin_edges, bin_counts, lbl = sn.histogram(mats=mats, num_bins=100)
    # Stairs -> histogram
    
    plt.stairs(bin_counts, bin_edges, fill=True, color='skyblue', edgecolor='black')
    plt.title(lbl)
    plt.xlabel("Spectral Norm")
    plt.ylabel("Number of Mats")
    plt.show()