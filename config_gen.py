#### sim_parameters
OUTPUT_FILE = "/home/luca/Uni/VI/Tesi/test_tg.csv"
N_EVGEN = 10000
N_TRK = 5
mu = 0.					 # precisione di allinemento (S. Coli)
sigma = 0.005			 # ALPIDE risoluzione spaziale 4-5 micron o 28/sqrt(12) 0.005
ProbNoise = 1e-6     # prendere il fake-rate ALPIDE 10^-10/ pixel------10^-6
ProbMiss = 0.05        # pixel morti 
fix_vertex = False