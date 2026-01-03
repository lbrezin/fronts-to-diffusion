"""
Provides the class and key functions to compute a non linear diffusion-nutrient model
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pnd
import h5py
from filelock import FileLock
import json
import hashlib
from datetime import datetime
import os
from scipy.linalg import solve_banded
from scipy.integrate import trapezoid
from scipy.stats import linregress


# Defining the class

class pd2:
    """Class that implements exponential growth, nonlinear biomass diffusion, and nutrient diffusion in two dimensions assuming rotational invariance."""
    def __init__(self, param):
        self.Db = param['Db'] # biomass diffusion
        self.Dn = param['Dn'] # nutrient diffusion
        self.alpha = param['alpha'] # Exponent for the nutrients
        self.beta = param['beta'] # Exponent for the biomass 
        self.g = param['gamma'] # consumption rate
        #self.dx = self.L/(self.N-1) # distance between points
        self.dx = param['accuracyX']*(self.Db*param['n0']/self.g)**0.5 #Defines dx as number of points in biomass front, which is set by accuracy (needs to be >1)
        #self.N = param['N'] # number of spatial points
        #self.L = self.N*self.dx # domain length
        self.x = np.arange(0,param['L'],self.dx) # locations of lattice points with dx spacing until L (excluded)
        self.N = len(self.x) #Number of points set by the arange function
        self.n_init = param['n0'] #Store initial nutrient concentration
        self.BC = param['BC'] # Paramater for the boundary condition. Can be 'Neu' for Neumann (no-flux) or 'Dir' for Dirichlet (fixed value)
        self.method = param['method'] #Parameter for the growth method. Can be 'heaviside' for growth rate independent of n for n>0, or 'linear' for linear in n
        self.t = param['t0'] # initial time
        self.n = param['n0']*np.ones(self.N, dtype=float) # initial nutrient concentration. User provides scalar n0 that gives the scale for nutrients
        self.b = np.zeros(self.N, dtype=float) # initial biomass concentration. Hard coded initial condition. Can be modified
        self.b[:10] = 0.1*param['n0'] #Set as a fraction of carrying capacity n0 on the left
        if param['dt'] > 0:
            self.dt = param['dt']
        else:
            self.dt = param['accuracyT']*min(0.5*np.max(self.n)/self.g,0.35*self.dx**2/self.Db/np.max(self.n)) # insure stability and accuracy. Could add a stability parameter.
        self.cnf = self.Dn*self.dt/2/self.dx**2 # factor in Crank-Nicolson
        self.bdf = self.Db*self.dt/self.dx**2 # factor in biomass diffusion
        self.gfo = self.g*self.dt/2 # used in growth update
        self.gft = 0.5*(self.g*self.dt/2) # used in growth update
        self.dm = np.zeros((3,self.N),dtype=float) # diffusion matrix for Crank-Nicolson
        # this is a banded matrix; see scipy.linalg.solve_banded
        # upper diagonal
        self.dm[0,2:] = -self.cnf*(np.ones(self.N-2) + self.dx/2/self.x[1:-1]) #Add the 1/r term in the diffusion
        # left reflective boundary condition
        self.dm[0,1] = -1 
        self.dm[1,0] = 1
        # main diagonal
        self.dm[1,1:-1] = 1 + 2*self.cnf*np.ones(self.N-2)
        # lower diagonal
        self.dm[2,:-2] = -self.cnf*(np.ones(self.N-2) - self.dx/2/self.x[1:-1])
        # right boundary condition
        #Dirchlet
        if self.BC == 'Dir':
            self.dm[2,-2] = 0
        #Neumann
        elif self.BC == 'Neu':
            self.dm[2,-2] = -1
        else:
            raise ValueError("Unsupported BC. Only 'Neu' and 'Dir' are currently supported")
        self.dm[1,-1] = 1
        if self.method == 'monod' or self.method == 'monod-heaviside':
            self.K = param['K']
        if self.method == 'maintenance' or self.method == 'maintenance-motility':
            self.K = param['K']
            self.nm = param['nm'] #Threshold of nutrient for maintenance cost
            
    def growth(self):
        "Implements growth update for dt/2 (for extra accuracy) using second order accurate Taylor's method."
        if self.method=='linear':
            update = self.gfo*self.n*self.b*(1 - self.gft*(self.b - self.n))
            self.n -= update
            self.b += update
        elif self.method=='heaviside':
            heav = (self.n > 10**(-6)).astype(int)
            update = self.gfo*heav*self.b
            self.n -= update
            self.b += update
        elif self.method == 'monod':
            update = self.gfo*self.n*self.b/(self.K*np.ones(self.N)+self.n)
            self.n -= update
            self.b += update
        elif self.method == 'maintenance' or self.method == 'maintenance-motility':
            heav = (self.n > self.nm).astype(int)
            updateb = self.gfo*self.n*self.b*heav
            updaten = self.gfo*self.n*self.b
            self.n -= updaten
            self.b += updateb
        else:
            update = self.gfo*self.n*self.b*(1 - self.gft*(self.b - self.n)) #Do linear by default
            self.n -= update
            self.b += update
        
        
    def biomass_diffusion(self):
        "Explicit implementation of biomass diffusion."
        if self.method == 'diffusive':
            self.b[1:-1] += self.bdf*(self.n[:-2]*np.power(self.b[:-2],2) - 2*self.n[1:-1]*np.power(self.b[1:-1],2) + self.n[2:]*np.power(self.b[2:],2)) # main update
            self.b[0] = self.b[1] # boundary conditions
            self.b[-1] = self.b[-2] # boundary conditions
        elif self.method == 'diffusive2':
            self.b[1:-1] += self.bdf*(np.power(self.n[:-2]*self.b[:-2],2) - 2*np.power(self.n[1:-1]*self.b[1:-1],2) + np.power(self.n[2:]*self.b[2:],2)) # main update
            self.b[0] = self.b[1] # boundary conditions
            self.b[-1] = self.b[-2] # boundary conditions
        elif self.method == 'diffusive3':
            #Left flux
            Jl = (np.power(self.b[:-2],self.beta) + np.power(self.b[1:-1],self.beta))/2*(self.b[1:-1]*self.n[1:-1]-self.b[:-2]*self.n[:-2])
            #Right flux
            Jr = (np.power(self.b[2:],self.beta) + np.power(self.b[1:-1],self.beta))/2*(self.b[2:]*self.n[2:]-self.b[1:-1]*self.n[1:-1])
            self.b[1:-1] += self.bdf*(Jr-Jl) # main update
            self.b[0] = self.b[1] # boundary conditions
            self.b[-1] = self.b[-2] # boundary conditions
        elif self.method == 'maintenance-motility':
            heav = (self.n > self.nm).astype(int)
            #Left flux
            Jl = ((np.power(self.n[:-2],self.alpha)*np.power(self.b[:-2],self.beta) + np.power(self.n[1:-1],self.alpha)*np.power(self.b[1:-1],self.beta))/2*(self.b[1:-1]-self.b[:-2]))*heav[1:-1]
            #Right flux
            Jr = ((np.power(self.n[2:],self.alpha)*np.power(self.b[2:],self.beta) + np.power(self.n[1:-1],self.alpha)*np.power(self.b[1:-1],self.beta))/2*(self.b[2:]-self.b[1:-1]))*heav[1:-1]
            self.b[1:-1] += self.bdf*(Jr-Jl) # main update
            self.b[0] = self.b[1] # boundary conditions
            self.b[-1] = self.b[-2] # boundary conditions
        else:
            #Left flux
            Jl = (self.x[:-2]*np.power(self.n[:-2],self.alpha)*np.power(self.b[:-2],self.beta) + self.x[1:-1]*np.power(self.n[1:-1],self.alpha)*np.power(self.b[1:-1],self.beta))/2*(self.b[1:-1]-self.b[:-2])
            #Right flux
            Jr = (self.x[2:]*np.power(self.n[2:],self.alpha)*np.power(self.b[2:],self.beta) + self.x[1:-1]*np.power(self.n[1:-1],self.alpha)*np.power(self.b[1:-1],self.beta))/2*(self.b[2:]-self.b[1:-1])
            self.b[1:-1] += self.bdf*(Jr-Jl)/self.x[1:-1] # main update
            self.b[0] = self.b[1] # boundary conditions
            self.b[-1] = self.b[-2] # boundary conditions
        
    def nutrient_diffusion(self):
        """Crank-Nicolson update."""
        rhs = self.n + self.cnf*(-2*self.n) # diagonal part
        rhs[0] = 0 # left boundary condition
        # right boundary condition
        #Dirichlet
        if self.BC == 'Dir':
            rhs[-1] = self.n_init # fixed initial nutrient concentration
        #Neumann
        elif self.BC == 'Neu':
            rhs[-1] = 0
        else:
            raise ValueError("Unsupported BC. Only 'Neu' and 'Dir' are currently supported")
        rhs[1:-1] += self.cnf*(self.n[2:]*(1+self.dx/2/self.x[1:-1])+self.n[:-2]*(1-self.dx/2/self.x[1:-1])) # bulk
        self.n = solve_banded((1,1), self.dm, rhs, overwrite_b=True)
    
    def step(self, n=1):
        """We use operator split method to advance by dt."""
        for i in range(n):
            self.biomass_diffusion()
            self.growth()
            self.nutrient_diffusion()
            self.growth()
            self.t += self.dt
        
    def plot(self):
        plt.plot(self.x, self.n, 'g', label='nutrient')
        plt.plot(self.x, self.b, 'b', label='biomass')
        plt.title('Time t={0:.2f}'.format(self.t))
        plt.xlabel('radial position, $r$')
        plt.ylabel('concentration')
        plt.legend()
        plt.grid()     



#Some saving functions
def param_hash(param):
    "Create a stable hash (order-insensitive) that provides a unique identifier to a set of parameters"
    return hashlib.sha1(json.dumps(param, sort_keys=True).encode()).hexdigest()

def timestamp():
    "Return current timestamp in compact sortable format. Will be apppended to param_hash to avoid overwriting in case code is modified"
    return datetime.now().strftime("%Y%m%dT%H%M%S")

def find_simulations_by_param(param, csvFile='lookup_table.csv',tol=1e-2):
    """Return summary DataFrame rows matching parameter set (by param_hash).
    Supports partial list of parameters, and returns all the simulations that have the subset of parameters.
    This supports float or integers matching, and allows some tolerance for 
    provided parameters, except for strings where the parameter has to be exact"""
    df = pnd.read_csv(csvFile)
    mask = np.ones(len(df), dtype=bool)
    for key, val in param.items():
        #Check that the parameter is actual one of the simulations
        if key not in df:
            raise ValueError(f"Parameter '{key}' is not a column in the summary table.")
        col = df[key]
        #Allow some tolerance if integer or float
        if np.issubdtype(col.dtype, np.floating) or isinstance(val, float) or np.issubdtype(col.dtype, np.integer):
            close = np.abs(col - val) <= tol
            #In case no value matches within the tolerance, still pick the closest thing
            if np.sum(close) == 0:
                idx_closest = np.argmin(np.abs(col - val))
                print(f" - No close match in column '{key}', using closest: {col.iloc[idx_closest]}")
                mask &= (col == col.iloc[idx_closest])
            #Otherwise, mark the search as true
            else:
                mask &= close
        #Strings have to match exactly
        else:
            mask &= (col == val)
    matches = df[mask]
    if len(matches) == 0:
        print(f"No runs found matching {query_param}")
    return matches  # This is a DataFrame with all matching rows
    
    
#Some analysis tools

def compute_front(b_array, dx, threshold):
    """ Computes the front position of an array (the biomass) due to some threshold. """
    b = np.asarray(b_array) # making sure we have an array
    indices = np.where(b >= threshold)[0] # finds all indices where the biomass is greater than or equal to the threshold

    if len(indices) == 0: # if the threshold is not meant it returns not a number
        print("Warning: No indices meet the threshold.")
        return np.nan

    i = indices[-1] # the first index where the biomass is greater than the threshold leading to the edge
    if i == 0: # if the first index is at teh boundary, then no linear interpolation can occur
        print("Warning: Index at boundary.")
        return 0.0 # if at boundary it is zero
    if i==len(b) - 1: # if the first index is at the boundary, then no linear interpolation can occur
        print("Warning: Index at end boundary.")
        return i*dx

    x0 = (i) * dx # the point before the threshold was crossed
    x1 = (i+1) * dx # the point where hte biomass first exceeds the threshold
    y0 = b[i] # find the biomass value around the threshold (before)
    y1 = b[i+1] # find the biomass value around the threshold (after)

    if y1 != y0: # checks if the biomass values differ, and if they do linear interpolation can occur
        frac = (threshold - y0) / (y1 - y0) # find where between x0 and x1 that the threshold is crossed
        res = x0 + frac * (x1 - x0)
        assert res>0, f"front index negative, position: {res}; before: {x0}; after: {x1}; frac: {frac}; index:{i}"
        return res # this computes hte exact position of the the front of the biomas that passes the threshold
    else: # if they do equal each other, no linear interpolation, so the upper position is returned
        return x1 # upper position
        # we do linear interpolation because we assume there is a linear relationship betwenn (x0, y0) and (x1, y1) which may not be the case with larger alphas but we shall see
        

def compute_front_nutrient(b_array, n_array, dx, threshold):
    """ Computes the nutrient density at where the biomass reaches the threshold value at the edge. """
    b = np.asarray(b_array) # making sure we have an array
    n = np.asarray(n_array) # making sure we have an array
    indices = np.where(b >= threshold)[0] # finds all indices where the biomass is greater than or equal to the threshold

    if len(indices) == 0: # if the threshold is not meant it returns not a number
        print("Warning: No indices meet the threshold.")
        return np.nan

    i = indices[-1] # the first index where the nutrients is greater than the threshold leading to the edge
    if i == 0: # if the first index is at teh boundary, then no linear interpolation can occur
        print("Warning: Index at boundary.")
        return 0.0 # if at boundary it is zero
    if i==len(b) - 1: # if the first index is at the boundary, then no linear interpolation can occur
        print("Warning: Index at end boundary.")
        return i*dx

    x0 = (i) * dx # the point before the threshold was crossed
    x1 = (i+1) * dx # the point where hte biomass first exceeds the threshold
    y0b = b[i] # find the biomass value around the threshold (before)
    y1b = b[i+1] # find the biomass value around the threshold (after)
    y0 = n[i] # find the nutrient value around the threshold (before)
    y1 = n[i+1] # find the nutrient value around the threshold (after)

    if y1b != y0b: # checks if the biomass values differ, and if they do linear interpolation can occur
        xc = x0+ (threshold - y0b) / (y1b - y0b)*(x1-x0) # find where between x0 and x1 that the threshold is crossed for the biomass
        return y0 + (y1 - y0)/(x1-x0)*(xc-x0) # Linear interpolation for the nutrient value at point where biomass is crossed
    else: # if they do equal each other, no linear interpolation, so the upper position is returned
        return y1 # upper position
        # we do linear interpolation because we assume there is a linear relationship betwenn (x0, y0) and (x1, y1) which may not be the case with larger alphas but we shall see



def collect_biomass_data(param, step=100,threshold_fraction=0.05):
    """Runs the simulation until nutrients are exhausted and returns total biomass as a function of time."""
    sim = pd2(param)
    threshold = threshold_fraction * sim.n_init 
    lt = [sim.t]
    ltb = [trapezoid(sim.b)*sim.dx]
    ltp = [compute_front(sim.b, sim.dx, threshold)]
    #Initialize condition
    condition = True
    while condition:
        #In Neumann case, stop when the nutrients at the end are being consumed
        if sim.BC == 'Neu':
            condition = sim.n[-1] > 0.999*sim.n_init
        #In Dirichlet case, stop when there is a gradient at the end
        elif sim.BC == 'Dir':
            condition = abs(sim.n[-1]-sim.n[-2])/sim.dx < 10**(-4)
        else:
            raise ValueError("Unsupported BC. Only 'Neu' and 'Dir' are currently supported")
        sim.step(step)
        lt.append(sim.t)
        ltb.append(trapezoid(2*np.pi*np.multiply(sim.b,sim.x),sim.x)) #Computes total biomass in 2d. 
        ltp.append(compute_front(sim.b, sim.dx, threshold)) # adds the new postion "point"

    return (lt, ltb, ltp)

def run_and_save(param, saveFile = 'results.h5', csvFile = 'lookup_table.csv', save_every = 10, step=100, threshold_fraction = 0.05):
    """Runs the simulation until nutrients are exhausted and saves the total biomass over time, the position of the front,
    and the biomass and nutrient every save_every*step timesteps. 
    Still returns the total biomass and front position at the end.""" 

    phash = param_hash(param) #creates the identifier
    ts = timestamp() #creates the time stamp when the simulation started
    group_id = f'{phash}_{ts}' #Creates the id for the simulation

    lockfile = saveFile + ".lock" #Creates the lock for file
    counter = 0 #Counts how many times the while loop is entered
    
    #Initialize objects to save densities
    b_snap = []
    n_snap = []
    t_snap = []
    
    #Runs simulation
    sim = pd2(param)
    threshold = threshold_fraction * sim.n_init 
    lt = [sim.t]
    ltb = [trapezoid(2*np.pi*np.multiply(sim.b,sim.x),sim.x)]
    ltp = [9*sim.dx] # creating the list of positions
    #Initialize condition
    condition = True
    while condition:
        #In Neumann case, stop when the nutrients at the end are being consumed
        if sim.BC == 'Neu':
            condition = sim.n[-1] > 0.999*sim.n_init
        #In Dirichlet case, stop when there is a gradient at the end
        elif sim.BC == 'Dir':
            condition = abs(sim.n[-1]-sim.n[-2])/sim.dx < 10**(-4)
        else:
            raise ValueError("Unsupported BC. Only 'Neu' and 'Dir' are currently supported")
        sim.step(step)
        #Time objects saved at every time points
        lt.append(sim.t)
        ltb.append(trapezoid(2*np.pi*np.multiply(sim.b,sim.x),sim.x))
        ltp.append(compute_front(sim.b, sim.dx, threshold)) # adds the new postion "point"
        #Save the densities
        if counter % save_every == 0: 
            #Copy arrays into the saved list. Used deepcopy just in case but we can use copy if slow since there are no nested structures yet. 
            b_snap.append(sim.b.copy()) 
            n_snap.append(sim.n.copy())
            t_snap.append(sim.t.copy())
        counter += 1
        
    #Save the simulation to HDF5 file while implementing a file lock so files don't write simultaneously
    with FileLock(lockfile):
        with h5py.File(saveFile, "a") as f:
            g = f.create_group(f'simulations/{group_id}')
            # Save parameter dict as JSON string
            g.attrs['param_json'] = json.dumps(param) #Saves the string of parameters
            g.attrs['param_hash'] = phash
            g.attrs['timestamp'] = ts
            g.create_dataset('biomass_snapshots', data=np.array(b_snap))
            g.create_dataset('position', data=np.array(sim.x))
            g.create_dataset('nutrient_snapshots', data=np.array(n_snap))
            g.create_dataset('time_snapshots', data=np.array(t_snap))
            g.create_dataset('time', data=np.array(lt))
            g.create_dataset('total_biomass', data=np.array(ltb))
            g.create_dataset('front_position', data=np.array(ltp))
    
        # Add/append to summary CSV
        param_row = param.copy()
        param_row['param_hash'] = phash
        param_row['timestamp'] = ts
        param_row['group_id'] = group_id
        df = pnd.DataFrame([param_row])
        with FileLock(csvFile + ".lock"):
            write_header = not os.path.exists(csvFile) #Write the header only if the 
            df.to_csv(csvFile, mode="a", header=write_header, index=False)
        print(f"Saved and merged simulation to group {group_id}")


def load_simulation_data(group_id, saveFile='results.h5'):
    "Once a group_id has been identified, loads the the results"
    with h5py.File(saveFile, 'r') as f:
        g = f[f'simulations/{group_id}']
        param = json.loads(g.attrs['param_json'])
        return {
            'param': param,
            'time': g['time'][()],
            'position': g['position'][()],
            'time_snapshots': g['time_snapshots'][()],
            'biomass_snapshots': g['biomass_snapshots'][()],
            'nutrient_snapshots': g['nutrient_snapshots'][()],
            'total_biomass': g['total_biomass'][()],
            'front_position': g['front_position'][()]
        }



def automatic_fit(time, biomass, r2_threshold=0.999999, min_len=2):
    """
    Finds the longest linear segment by shrinking from both ends.
    Tracks best-fit segment even if R² threshold is never met.
    
    Returns:
        slope, intercept, (start_idx, end_idx)
    """
    n = len(time)
    start = 0
    end = n

    best_r2 = -np.inf
    best_fit = None
    best_segment = (0, n)

    while (end - start) >= min_len:
        t_seg = time[start:end]
        b_seg = biomass[start:end]
        slope, intercept, r_value, _, _ = linregress(t_seg, b_seg)
        r2 = r_value ** 2

        # Track best segment seen so far
        if r2 > best_r2 and (end - start) >= min_len:
            best_r2 = r2
            best_fit = (slope, intercept)
            best_segment = (start, end)

        # Stop early if good enough
        if r2 >= r2_threshold:
            return slope, intercept, (start, end), r2

        # Evaluate shrinking from front and back
        shrink_front_r2 = -np.inf
        if (end - (start + 1)) >= min_len:
            shrink_front_r2 = linregress(time[start+1:end], biomass[start+1:end])[2] ** 2
        shrink_back_r2 = -np.inf
        if ((end - 1) - start) >= min_len:
            shrink_back_r2 = linregress(time[start:end-1], biomass[start:end-1])[2] ** 2
#
        # Choose direction that improves R² the most
        if shrink_front_r2 >= shrink_back_r2:
            start += 1
        else:
            end -= 1
        #start += 1
        #end -= 1

    print("Warning: no segment met R² threshold. Returning best segment found.")
    return best_fit[0], best_fit[1], best_segment, r2

def linearfit_dataset(data, min_len = 0.5):
    #Performs a linear fit of the biomass data from the dataset. This can then be used to compute the velocity. min_len represents fraction of biomass data as minimum to fit
    fit=[]
    for index,elem in enumerate(data):
        time = elem['time']
        biomass = elem['total_biomass']
        slope, intercept, (start, end), r2 = automatic_fit(time, biomass, r2_threshold=0.8, min_len=int(min_len*len(biomass)))
        results = {
            "velocity" : slope,
            "r2" : r2
        }
        fit.append(results)
    return fit

def quadraticfit_dataset(data, min_len = 0.5):
    #Performs a linear fit of the biomass^2 data from the dataset. This can then be used to compute the K coefficient. min_len represents fraction of biomass data as minimum to fit
    fit=[]
    for index,elem in enumerate(data):
        time = elem['time']
        biomass = elem['total_biomass']
        slope, intercept, (start, end), r2 = automatic_fit(time, np.power(biomass,2), r2_threshold=0.8, min_len=int(min_len*len(biomass)))
        results = {
            "K" : slope,
            "r2" : r2
        }
        fit.append(results)
    return fit
    
def sqrtfit_dataset(data, min_len = 0.5):
    #Performs a linear fit of the biomass data from the dataset. This can then be used to compute the velocity. min_len represents fraction of biomass data as minimum to fit
    fit=[]
    for index,elem in enumerate(data):
        time = elem['time']
        biomass = elem['total_biomass']
        slope, intercept, (start, end), r2 = automatic_fit(np.power(time,2), biomass, r2_threshold=0.8, min_len=int(min_len*len(biomass)))
        results = {
            "xi" : slope,
            "r2" : r2
        }
        fit.append(results)
    return fit

def sqrtedgefit_dataset(data, min_len = 0.5):
    #Performs a linear fit of the biomass data from the dataset. This can then be used to compute the velocity. min_len represents fraction of biomass data as minimum to fit
    fit=[]
    for index,elem in enumerate(data):
        time = elem['time']
        edge = elem['front_position']
        slope, intercept, (start, end), r2 = automatic_fit(np.power(time,0.5), edge, r2_threshold=0.8, min_len=int(min_len*len(edge)))
        results = {
            "xi" : slope,
            "r2" : r2
        }
        fit.append(results)
    return fit

def edgefit_dataset(data, min_len = 0.5,r2_threshold = 0.8,exclude=3):
    #Performs a linear fit of the biomass data from the dataset. This can then be used to compute the velocity. min_len represents fraction of biomass data as minimum to fit
    fit=[]
    for index,elem in enumerate(data):
        time = elem['time'][:-exclude]
        edge = elem['front_position'][:-exclude]
        slope, intercept, (start, end), r2 = automatic_fit(np.power(time,0.5), edge, r2_threshold = r2_threshold, min_len=int(min_len*len(edge)))
        results = {
            "xi" : slope,
            "r2" : r2
        }
        fit.append(results)
    return fit

def linear_edgefit_dataset(data, min_len = 0.5,r2_threshold = 0.8,exclude=3):
    #Performs a linear fit of the biomass data from the dataset. This can then be used to compute the velocity. min_len represents fraction of biomass data as minimum to fit
    fit=[]
    for index,elem in enumerate(data):
        time = elem['time'][:-exclude]
        edge = elem['front_position'][:-exclude]
        slope, intercept, (start, end), r2 = automatic_fit(time, edge, r2_threshold=r2_threshold, min_len=int(min_len*len(edge)))
        results = {
            "xi" : slope,
            "r2" : r2
        }
        fit.append(results)
    return fit
            
