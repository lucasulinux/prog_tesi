import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from skspatial.objects import Line, Points
import config_gen
import pandas as pd
from itertools import product

#### sensor_parameters
dist_z = 8.5 # [mm] distance between planes
Lgap = 4.85 # GAP STAVE mm + addition distance for space compliance (mechanincal)
Sgap = 0.15 # GAP CHIP mm
width = 30 #width of the turrets [mm]
stavew=width+Sgap
ProbNoise = config_gen.ProbNoise

def generate_random_line(point=0,randompoint=True):
    '''
        funzione che genera una retta casuale, nei bound richiesti (da verificare se i punti poi apparterranno al mio sensore)
        se si volesse inserire il vertice della retta, inserirlo in point come np.array, settare randompoint=True
        DA VERIFICARE SE FUNZIONANO LE MODIFICHE
    '''
    if(randompoint==True):
         # Generate a random point on the line
        point0_on_line = np.append(np.random.rand(2),0) #Random point coordinates
        point0_on_line[0] *= 170
        point0_on_line[1] *= 150
        
    else:
        point0_on_line = point 

    point1_on_line = np.append(np.random.rand(2),-17)  # Random point coordinates
    point1_on_line[0] *= 170
    point1_on_line[1] *= 150
    
    direction = (point1_on_line - point0_on_line) / np.linalg.norm(point1_on_line - point0_on_line)
    
    return point0_on_line, direction

def get_points(vec_reco,cent_reco, zpos,rum=True):
    '''
    funzione che ottiene i punti che hittano il mio rilevatore, prendendoli da una retta data
    restituisce una matrice np con tre vettori con le coordinate dei punti di hit
    '''
    x = np.zeros_like(zpos)
    y = np.zeros_like(zpos)
    vec_reco = vec_reco

    for ilay, z in enumerate(zpos):
        x[ilay] = zpos[ilay]*vec_reco[0]/(vec_reco[2]) - (cent_reco[2]*(vec_reco[0]/vec_reco[2])-cent_reco[0])
        y[ilay] = zpos[ilay]*vec_reco[1]/(vec_reco[2]) - (cent_reco[2]*(vec_reco[1]/vec_reco[2])-cent_reco[1])
        
        if (rum==True):
            bias_x, bias_y = apply_noise(x[ilay],y[ilay])
            x[ilay] = bias_x
            y[ilay] = bias_y

    line_points = np.stack([x,y,zpos]).T
    return line_points

import config_gen

mu = config_gen.mu
sigma = config_gen.sigma

def apply_noise(x,y):
    bias_x = x + np.random.normal(mu,sigma)
    bias_y = y + np.random.normal(mu,sigma)
    return bias_x,bias_y

def limiti_spaziali(points):
    ''' 
    Verifica che i punti generati siano negli estremi spaziali del mio rilevatore
    '''
    x = points.T[0]
    y = points.T[1]
    return all(x >= 0) and all(x <= 150) and all(y >= 0) and all(y <= 175)
def missed_stave(i, stavew, Lgap):
    ranges = [
        (stavew, stavew + Lgap),
        (2 * stavew + Lgap, 2 * (stavew + Lgap)),
        (stavew + 2 * (stavew + Lgap), 3 * (stavew + Lgap)),
        (stavew + 3 * (stavew + Lgap), 4 * (stavew + Lgap)),
    ]
    return any(start < i < end for start, end in ranges)

class dati:
    def __init__( self, indexev, points , direction , pointonline, idnumber ):
        
        self.indexev = indexev #identifies the plot of origin, the event of origin
        self.points = points
        self.direction = direction
        self.pol = pointonline
        self.id = idnumber #identifies if the point is a noise point (-1) or a point of the line (1,2,3)

def noise_gen(ProbNoise, dist_z):
    '''
        Generates noisy pixels in the three planes 
        (maximum of one for plane, maybe add the option of multiple points in every plane)
        return a list of points
    '''
    noise=[]
    j = 0
    for i in range(3):
        if np.random.uniform(0, 1) < ProbNoise:
            x = np.random.uniform(0, 150)
            y = np.random.uniform(0, 30)
            z = -dist_z * i
            if(len(noise) == 0):
                noise = np.array([x, y, z])
            else:
                noise = np.concatenate( [noise,np.array([x, y, z])] , axis = 0 )
    if(len(noise) != 0):
        noise_matrix = noise.reshape(-1,3)
    else:
        noise_matrix = noise
    
    return noise_matrix
def setup_3d_track_plot():
    fig = plt.figure(figsize = (20,8))
    gs = GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, :2])
    plt.title('x-z view',fontsize = 15, loc='left')
    
    ax1.plot([0.,150.],[0.,0.], color = 'blue')
    ax1.plot([0.,150.],[-1*dist_z,-1*dist_z], color = 'blue')
    ax1.plot([0.,150.],[-2*dist_z,-2*dist_z], color = 'blue')
    ax1.set_xlabel('x [mm]')
    ax1.set_ylabel('z [mm]')
    ax1.grid(alpha = 0.5)
    ax1.set_xlim(-10,160)
    ax1.set_ylim(-25.5,8.5)

    ax2 = fig.add_subplot(gs[1, :2])
    
    plt.title('y-z view',fontsize = 15, loc='left')
    

    for tur in range(0,5):
        shift = tur*(30+Lgap+Sgap)
        ax2.plot([shift + 0.,shift + 30.15],[0.,0.], color = 'blue')
        ax2.plot([shift + 0.,shift + 30.15],[-1*dist_z,-1*dist_z], color = 'blue')
        ax2.plot([shift + 0.,shift + 30.15],[-2*dist_z,-2*dist_z], color = 'blue')
    ax2.set_xlabel('y [mm]')
    ax2.set_ylabel('z [mm]')
    ax2.grid(alpha = 0.5)
    ax2.set_xlim(-10,180)
    ax2.set_ylim(-25.5,8.5)

    ax3 = fig.add_subplot(gs[:, 2])
    plt.title('x-y view',fontsize = 15)

    for tur in range(0,5):
        shift = tur*(30+Lgap+Sgap)
        ax3.plot([0.,150.],[shift + 0.,shift + 0.], color = 'blue')
        ax3.plot([0.,150.],[shift + 30.15,shift + 30.15], color = 'blue')
        ax3.plot([0.,0.],[shift + 0.,shift + 30.15], color = 'blue')
        ax3.plot([150.,150.],[shift + 0.,shift + 30.15], color = 'blue')

    ax3.set_xlabel('x [mm]')
    ax3.set_ylabel('y [mm]')
    ax3.grid(alpha = 0.5)
    ax3.set_xlim(-10,160)
    ax3.set_ylim(-15,180)
    return fig, ax1, ax2, ax3

def display_single_fit(point,vec_reco,cent_reco,seg=True):
     fig, ax1, ax2, ax3 = setup_3d_track_plot()
     for i in range(len(point)):
          x_pos,y_pos,z_pos = point[i][:,0], point[i][:,1], point[i][:,2]
          color = plt.cm.tab10(i % 10)
          ax1.scatter(x_pos,z_pos, color = color, label = 'track', alpha = 0.5, s = 100)
          ax2.scatter(y_pos,z_pos, color = color, label = 'track', alpha = 0.5, s = 100)     
          ax3.scatter(x_pos,y_pos, color = color, label = 'track', alpha = 0.5, s = 100)
          if (seg==True) or (is_null(vec_reco[i])==False):
            ax1.plot(get_points(vec_reco[i], cent_reco[i], np.linspace(-30, 5), False)[:, 0],
            np.linspace(-30, 5), color=color, label=f'Track {i+1}')
            ax2.plot(get_points(vec_reco[i], cent_reco[i], np.linspace(-30, 5), False)[:, 1],
            np.linspace(-30, 5), color=color, label=f'Track {i+1}')
            ax3.plot(get_points(vec_reco[i], cent_reco[i], np.linspace(-30, 5), False)[:, 0],
            get_points(vec_reco[i], cent_reco[i], np.linspace(-30, 5), False)[:, 1], color=color, label=f'Track {i+1}')

def fit_straight_line(points):
    # Convert points to skspatial Points object
    sk_points = Points(np.array(points))

    # Fit a line using the points
    line = Line.best_fit(sk_points)
    dir_reco = np.array(line.direction)
    centroid_reco = np.array(line.point)

    return dir_reco,centroid_reco 

outfile = config_gen.OUTPUT_FILE

def save_csv_py(data):
    ''' 
        in input una classe di tipo data
    '''
    x_pos,y_pos,z_pos,trk_id,event=[],[],[],[],[]
    for i in range(len(data)):
        if(len(data[i].points)>0):
            x_pos.append(data[i].points[:,0])
            y_pos.append(data[i].points[:,1])
            z_pos.append(data[i].points[:,2])
            trk_id.append(data[i].id*np.ones_like(data[i].points[:,0]))
            event.append(data[i].indexev*np.ones_like(data[i].points[:,0]))
    x=np.concatenate(x_pos)
    y=np.concatenate(y_pos)
    z=np.concatenate(z_pos)
    id=np.concatenate(trk_id)
    ev=np.concatenate(event)
    d = {"event": ev ,"x_pos": x ,"y_pos": y ,"z_pos": z ,"trk_index": id}
    df = pd.DataFrame(d)
    df['event'] = df['event'].astype(int)
    df['trk_index'] = df['trk_index'].astype(int)
    df.to_csv(outfile, index=False)


def read_csv_py(filename):
    df = pd.read_csv(filename)
    x = df['x_pos']
    y = df['y_pos']
    z = df['z_pos']
    ev = df['event'].astype(int)
    trk = df['trk_index'].astype(int)

    return x,y,z,ev,trk

def data_t_gen(x, y, z, ev, trk):
    '''
        Prende dei dati con tre coordinate spaziali, un indice di evento (numero di generazione) e un indice di appartenenza alla traccia (trk)
        Genera una lista di oggetti di tipo dati, con diverse entrate per le diverse tracce di appartenenza
    '''
    cont = []
    df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'ev': ev, 'trk': trk})
    grouped = df.groupby(['ev', 'trk'])
    
    for (event_id, track_id), group in grouped:
        points = group[['x', 'y', 'z']].values
        if len(points) > 0:
            cont.append(dati(event_id, points, 0, 0, track_id))
    return cont

def chi2_line(points):
    """
    Calcola la somma dei quadrati delle distanze ortogonali dei punti dalla retta best-fit.
    """
    line = Line.best_fit(points)
    direction = line.direction / np.linalg.norm(line.direction)
    point_on_line = line.point
    diff = points - point_on_line
    cross = np.cross(diff, direction)
    distances = np.linalg.norm(cross, axis=1)
    chi2 = np.sum(distances**2/sigma**2)
    return chi2

def get_combinations(points, dist_z):
    """
    Restituisce tutte le combinazioni di punti validi tra i diversi livelli z.
    """
    # Trova i livelli z unici presenti nei punti
    livelli_z = np.unique(points[:, 2])
    # Costruisci un dizionario {z: punti_a_quel_livello}
    punti_per_livello = {z: np.array([p for p in points if int(p[2]) == int(z) and not is_null(p)]) for z in livelli_z}
    # Prendi solo i livelli che hanno almeno un punto valido
    livelli_validi = [z for z in punti_per_livello if len(punti_per_livello[z]) > 0]
    # Se meno di 2 livelli, nessuna combinazione possibile
    if len(livelli_validi) < 2:
        return np.array([])
    # Se esattamente 2 livelli, prodotto cartesiano tra i due
    if len(livelli_validi) == 2:
        return np.array(list(product(punti_per_livello[livelli_validi[0]], punti_per_livello[livelli_validi[1]])))
    # Se 3 o più livelli, prodotto cartesiano tra i primi 3 (o tutti)
    else:
        return np.array(list(product(*[punti_per_livello[z] for z in livelli_validi[:3]])))



def is_null(point):
    return np.allclose(point, 0)
        