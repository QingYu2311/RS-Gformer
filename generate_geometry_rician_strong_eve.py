import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import time,os,yaml

def generate_positions_in_annulus(K,d_min,d_max):
    theta = 2*np.pi*np.random.rand(K).astype(np.float32)
    u = np.random.rand(K).astype(np.float32)
    r = np.sqrt(d_min**2 + u*(d_max**2-d_min**2)).astype(np.float32)

    x = r*np.cos(theta)
    y = r*np.sin(theta)

    positions = np.stack([x,y],axis=-1).astype(np.float32)
    return positions,r

def generate_eve_positions_near_lu(lu_positions,Eve,d_near_min,d_near_max):
    """Place each Eve near its paired LU (cyclic pairing when J > K)."""
    K = lu_positions.shape[0]
    eve_positions = np.zeros([Eve,2],dtype=np.float32)
    paired_lu_index = np.zeros([Eve],dtype=np.int64)

    for j in range(Eve):
        k = j%K
        paired_lu_index[j] = k

        theta = 2*np.pi*np.random.rand()
        u = np.random.rand()
        r = np.sqrt(d_near_min**2 + u*(d_near_max**2-d_near_min**2))

        offset = np.array([
            r*np.cos(theta),
            r*np.sin(theta)
        ],dtype=np.float32)

        eve_positions[j] = lu_positions[k]+offset

    distance_to_bs = np.sqrt(np.sum(eve_positions**2,axis=-1)).astype(np.float32)
    distance_to_paired_lu = np.sqrt(
        np.sum((eve_positions-lu_positions[paired_lu_index])**2,axis=-1)
    ).astype(np.float32)

    return eve_positions,distance_to_bs,distance_to_paired_lu,paired_lu_index

def load_or_create_lu_position_dataset(number,K,d_min,d_max,file_path):
    """Reuse a saved LU position dataset, or create it once."""
    if os.path.exists(file_path):
        saved_data = torch.load(file_path,map_location='cpu')

        positions = saved_data['positions']
        distance_m = saved_data['distance_m']

        if saved_data['number'] != number or saved_data['K'] != K:
            raise ValueError(
                f"Existing LU position dataset mismatch: number={saved_data['number']}, K={saved_data['K']}"
            )

        print(f"Reusing existing LU positions: {file_path}")
        return positions,distance_m

    start_time = time.time()
    positions = np.zeros([number,K,2],dtype=np.float32)
    distance_m = np.zeros([number,K],dtype=np.float32)

    for i in range(number):
        pos_temp,dis_temp = generate_positions_in_annulus(
            K,d_min,d_max
        )

        positions[i,:,:] = pos_temp
        distance_m[i,:] = dis_temp

        if (i+1)%2000==0:
            elapsed = time.time()-start_time
            print(f"Generated {i+1}/{number} LU position samples, elapsed: {elapsed:.2f}s",flush=True)

    positions = torch.FloatTensor(positions)
    distance_m = torch.FloatTensor(distance_m)

    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    torch.save({
        'positions':positions,
        'distance_m':distance_m,
        'number':number,
        'K':K,
        'd_min':d_min,
        'd_max':d_max,
        'scenario_name':'exp01c_geo_rician_strong_eve'
    },file_path)

    print(f"LU position dataset saved to: {file_path}")
    return positions,distance_m

def load_or_create_eve_position_dataset(
    lu_positions,Eve,d_near_min,d_near_max,file_path
):
    """Reuse saved Eve positions, or generate them from fixed LU positions."""
    number = lu_positions.shape[0]
    K = lu_positions.shape[1]

    if os.path.exists(file_path):
        saved_data = torch.load(file_path,map_location='cpu')

        positions = saved_data['positions']
        distance_m = saved_data['distance_m']
        distance_to_paired_lu = saved_data['distance_to_paired_lu']
        paired_lu_index = saved_data['paired_lu_index']

        if saved_data['number'] != number or saved_data['Eve'] != Eve or saved_data['K'] != K:
            raise ValueError(
                f"Existing Eve position dataset mismatch: number={saved_data['number']}, Eve={saved_data['Eve']}, K={saved_data['K']}"
            )

        print(f"Reusing existing Eve positions: {file_path}")
        return positions,distance_m,distance_to_paired_lu,paired_lu_index

    start_time = time.time()
    positions = np.zeros([number,Eve,2],dtype=np.float32)
    distance_m = np.zeros([number,Eve],dtype=np.float32)
    distance_to_paired_lu = np.zeros([number,Eve],dtype=np.float32)
    paired_lu_index = np.zeros([number,Eve],dtype=np.int64)

    lu_positions_np = lu_positions.cpu().numpy() if torch.is_tensor(lu_positions) else lu_positions

    for i in range(number):
        eve_pos,eve_dis,eve_to_lu,pair_idx = generate_eve_positions_near_lu(
            lu_positions_np[i],Eve,d_near_min,d_near_max
        )

        positions[i,:,:] = eve_pos
        distance_m[i,:] = eve_dis
        distance_to_paired_lu[i,:] = eve_to_lu
        paired_lu_index[i,:] = pair_idx

        if (i+1)%2000==0:
            elapsed = time.time()-start_time
            print(f"Generated {i+1}/{number} strong-Eve position samples, elapsed: {elapsed:.2f}s",flush=True)

    positions = torch.FloatTensor(positions)
    distance_m = torch.FloatTensor(distance_m)
    distance_to_paired_lu = torch.FloatTensor(distance_to_paired_lu)
    paired_lu_index = torch.LongTensor(paired_lu_index)

    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    torch.save({
        'positions':positions,
        'distance_m':distance_m,
        'distance_to_paired_lu':distance_to_paired_lu,
        'paired_lu_index':paired_lu_index,
        'number':number,
        'K':K,
        'Eve':Eve,
        'd_near_min':d_near_min,
        'd_near_max':d_near_max,
        'scenario_name':'exp01c_geo_rician_strong_eve'
    },file_path)

    print(f"Eve position dataset saved to: {file_path}")
    return positions,distance_m,distance_to_paired_lu,paired_lu_index

def calculate_pathloss_db(distance_m):
    return np.float32(0.0)

def generate_ula_los_from_positions(positions,NT,fc_hz=3.5e9):
    """Generate ULA LoS steering vectors with a distance-dependent phase."""
    c = 3e8
    wavelength = c/fc_hz

    x = positions[:,0].astype(np.float64)
    y = positions[:,1].astype(np.float64)

    distance = np.sqrt(x**2+y**2)
    theta = np.arctan2(y,x)

    antenna_index = np.arange(NT,dtype=np.float64)[None,:]

    steering = np.exp(
        -1j*np.pi*antenna_index*np.sin(theta)[:,None]
    )

    common_phase = np.exp(
        -1j*2*np.pi*distance/wavelength
    )[:,None]

    H_LOS = common_phase*steering
    return H_LOS.astype(np.complex64)

def generate_H_rician_from_positions(positions,NT,rician_K_dB,fc_hz=3.5e9):
    distance_m = np.sqrt(np.sum(positions**2,axis=-1)).astype(np.float32)
    pathloss_db = calculate_pathloss_db(distance_m)

    amplitude = np.power(10.0,-pathloss_db/20.0).astype(np.float32)

    K_R = 10**(rician_K_dB/10.0)

    H_LOS = generate_ula_los_from_positions(
        positions,NT,fc_hz=fc_hz
    )

    K = positions.shape[0]
    H_NLOS = np.sqrt(1/2)*(
        np.random.randn(K,NT).astype(np.float32)
        +1j*np.random.randn(K,NT).astype(np.float32)
    )

    H_small = (
        np.sqrt(K_R/(K_R+1.0))*H_LOS
        +np.sqrt(1.0/(K_R+1.0))*H_NLOS
    )

    H = amplitude[:,None]*H_small

    return H.astype(np.complex64),distance_m,pathloss_db

class Generate_H_geometry_rician_dataset(Dataset):
    def __init__(
        self,number,K,NT,d_min,d_max,
        rician_K_dB,fc_hz=3.5e9,file_path=None,fixed_positions=None
    ):
        super().__init__()
        start_time = time.time()

        self.number = number
        self.K = K
        self.NT = NT
        self.d_min = d_min
        self.d_max = d_max
        self.rician_K_dB = rician_K_dB
        self.fc_hz = fc_hz
        self.file_path = file_path
        self.scenario_name = 'exp01c_geo_rician_strong_eve'

        self.data = np.zeros([number,2,K,NT],dtype=np.float32)
        self.positions = np.zeros([number,K,2],dtype=np.float32)
        self.distance_m = np.zeros([number,K],dtype=np.float32)
        self.pathloss_db = np.zeros([number,K],dtype=np.float32)

        fixed_positions_np = None
        if fixed_positions is not None:
            fixed_positions_np = fixed_positions.cpu().numpy() if torch.is_tensor(fixed_positions) else fixed_positions
            if fixed_positions_np.shape != (number,K,2):
                raise ValueError(
                    f"Expected fixed_positions shape {(number,K,2)}; got {fixed_positions_np.shape}"
                )

        for i in range(0,self.number):
            if fixed_positions_np is None:
                positions,distance_m = generate_positions_in_annulus(
                    K,d_min,d_max
                )
            else:
                positions = fixed_positions_np[i].astype(np.float32)
                distance_m = np.sqrt(np.sum(positions**2,axis=-1)).astype(np.float32)

            H_temp,dis_temp,pl_temp = generate_H_rician_from_positions(
                positions,NT,rician_K_dB,fc_hz=fc_hz
            )

            self.data[i,0,:,:] = np.real(H_temp)
            self.data[i,1,:,:] = np.imag(H_temp)
            self.positions[i,:,:] = positions
            self.distance_m[i,:] = dis_temp
            self.pathloss_db[i,:] = pl_temp

            if (i+1)%2000==0:
                elapsed = time.time()-start_time
                print(f"Generated {i+1}/{number} samples, elapsed: {elapsed:.2f}s",flush=True)

        self.data = torch.FloatTensor(np.array(self.data))
        self.positions = torch.FloatTensor(np.array(self.positions))
        self.distance_m = torch.FloatTensor(np.array(self.distance_m))
        self.pathloss_db = torch.FloatTensor(np.array(self.pathloss_db))

        print(f"Dataset generated, elapsed: {time.time()-start_time:.2f}s")

        if file_path:
            self.save_dataset_to_filepath()

    def save_dataset_to_filepath(self):
        os.makedirs(os.path.dirname(self.file_path),exist_ok=True)

        save_dict = {
            'data':self.data,
            'positions':self.positions,
            'distance_m':self.distance_m,
            'pathloss_db':self.pathloss_db,
            'K':self.K,
            'NT':self.NT,
            'number':self.number,
            'd_min':self.d_min,
            'd_max':self.d_max,
            'rician_K_dB':self.rician_K_dB,
            'fc_hz':self.fc_hz,
            'scenario_name':self.scenario_name
        }

        if hasattr(self,'mu'):
            save_dict['mu'] = self.mu
        if hasattr(self,'distance_to_paired_lu'):
            save_dict['distance_to_paired_lu'] = self.distance_to_paired_lu
        if hasattr(self,'paired_lu_index'):
            save_dict['paired_lu_index'] = self.paired_lu_index

        torch.save(save_dict,self.file_path)
        print(f"Dataset saved to: {self.file_path}")

    def __len__(self):
        return self.number

    def __getitem__(self,index):
        return self.data[index]

def load_dataset(filename):
    """Load a Geometry + Rician dataset from disk."""
    saved_data = torch.load(filename,map_location='cpu')

    dataset = Generate_H_geometry_rician_dataset.__new__(Generate_H_geometry_rician_dataset)
    dataset.data = saved_data['data']
    dataset.positions = saved_data.get('positions',None)
    dataset.distance_m = saved_data.get('distance_m',None)
    dataset.pathloss_db = saved_data.get('pathloss_db',None)
    dataset.K = saved_data['K']
    dataset.NT = saved_data['NT']
    dataset.number = saved_data['number']
    dataset.d_min = saved_data.get('d_min',None)
    dataset.d_max = saved_data.get('d_max',None)
    dataset.rician_K_dB = saved_data.get('rician_K_dB',None)
    dataset.fc_hz = saved_data.get('fc_hz',None)
    dataset.scenario_name = saved_data.get('scenario_name','exp01c_geo_rician_strong_eve')
    dataset.file_path = filename

    if 'mu' in saved_data:
        dataset.mu = saved_data['mu']
    if 'distance_to_paired_lu' in saved_data:
        dataset.distance_to_paired_lu = saved_data['distance_to_paired_lu']
    if 'paired_lu_index' in saved_data:
        dataset.paired_lu_index = saved_data['paired_lu_index']

    print(f"Dataset loaded from {filename}")
    print(f"Dataset: K={dataset.K}, NT={dataset.NT}, samples={dataset.number}")
    return dataset

def load_dataset_if_compatible(
    filename,K,NT,number,rician_K_dB,fc_hz,
    fixed_positions=None,mu=None
):
    """Reuse a compatible dataset; otherwise regenerate channel samples."""
    if not os.path.exists(filename):
        return None

    dataset = load_dataset(filename)

    is_compatible = True
    is_compatible = is_compatible and dataset.K == K
    is_compatible = is_compatible and dataset.NT == NT
    is_compatible = is_compatible and dataset.number == number

    if dataset.rician_K_dB is not None:
        is_compatible = is_compatible and abs(float(dataset.rician_K_dB)-float(rician_K_dB))<1e-12
    if dataset.fc_hz is not None:
        is_compatible = is_compatible and abs(float(dataset.fc_hz)-float(fc_hz))<1e-6

    if fixed_positions is not None and dataset.positions is not None:
        fixed_positions_cpu = fixed_positions.cpu() if torch.is_tensor(fixed_positions) else torch.FloatTensor(fixed_positions)
        is_compatible = is_compatible and dataset.positions.shape == fixed_positions_cpu.shape
        if is_compatible:
            is_compatible = torch.equal(dataset.positions,fixed_positions_cpu)

    if mu is not None:
        is_compatible = is_compatible and hasattr(dataset,'mu')
        if hasattr(dataset,'mu'):
            is_compatible = is_compatible and abs(float(dataset.mu)-float(mu))<1e-12

    if is_compatible:
        print(f"Reusing compatible channel dataset: {filename}")
        return dataset

    print(f"Regenerating channels; reusing fixed positions: {filename}")
    return None

def construct_strong_eve_true_dataset(
    dataset_LU,Eve,d_near_min,d_near_max,
    rician_K_dB,fc_hz,save_path=None,
    fixed_eve_positions=None,fixed_distance_to_paired_lu=None,fixed_paired_lu_index=None
):
    """Generate true Eve channels from paired-LU geometry and Rician fading."""
    start_time = time.time()

    dataset_eve_true = Generate_H_geometry_rician_dataset.__new__(Generate_H_geometry_rician_dataset)
    dataset_eve_true.number = dataset_LU.number
    dataset_eve_true.K = Eve
    dataset_eve_true.NT = dataset_LU.NT
    dataset_eve_true.d_min = None
    dataset_eve_true.d_max = None
    dataset_eve_true.rician_K_dB = rician_K_dB
    dataset_eve_true.fc_hz = fc_hz
    dataset_eve_true.file_path = save_path
    dataset_eve_true.scenario_name = 'exp01c_geo_rician_strong_eve'

    number = dataset_LU.number
    NT = dataset_LU.NT

    data = np.zeros([number,2,Eve,NT],dtype=np.float32)
    positions = np.zeros([number,Eve,2],dtype=np.float32)
    distance_m = np.zeros([number,Eve],dtype=np.float32)
    pathloss_db = np.zeros([number,Eve],dtype=np.float32)
    distance_to_paired_lu = np.zeros([number,Eve],dtype=np.float32)
    paired_lu_index = np.zeros([number,Eve],dtype=np.int64)

    lu_positions_np = dataset_LU.positions.cpu().numpy()

    fixed_eve_positions_np = None
    fixed_distance_to_paired_lu_np = None
    fixed_paired_lu_index_np = None

    if fixed_eve_positions is not None:
        fixed_eve_positions_np = fixed_eve_positions.cpu().numpy() if torch.is_tensor(fixed_eve_positions) else fixed_eve_positions
    if fixed_distance_to_paired_lu is not None:
        fixed_distance_to_paired_lu_np = fixed_distance_to_paired_lu.cpu().numpy() if torch.is_tensor(fixed_distance_to_paired_lu) else fixed_distance_to_paired_lu
    if fixed_paired_lu_index is not None:
        fixed_paired_lu_index_np = fixed_paired_lu_index.cpu().numpy() if torch.is_tensor(fixed_paired_lu_index) else fixed_paired_lu_index

    for i in range(number):
        if fixed_eve_positions_np is None:
            eve_pos,eve_dis,eve_to_lu,pair_idx = generate_eve_positions_near_lu(
                lu_positions_np[i],Eve,d_near_min,d_near_max
            )
        else:
            eve_pos = fixed_eve_positions_np[i].astype(np.float32)
            eve_dis = np.sqrt(np.sum(eve_pos**2,axis=-1)).astype(np.float32)

            if fixed_paired_lu_index_np is None:
                pair_idx = np.arange(Eve,dtype=np.int64)%dataset_LU.K
            else:
                pair_idx = fixed_paired_lu_index_np[i].astype(np.int64)

            if fixed_distance_to_paired_lu_np is None:
                eve_to_lu = np.sqrt(
                    np.sum((eve_pos-lu_positions_np[i][pair_idx])**2,axis=-1)
                ).astype(np.float32)
            else:
                eve_to_lu = fixed_distance_to_paired_lu_np[i].astype(np.float32)

        H_temp,dis_temp,pl_temp = generate_H_rician_from_positions(
            eve_pos,NT,rician_K_dB,fc_hz=fc_hz
        )

        data[i,0,:,:] = np.real(H_temp)
        data[i,1,:,:] = np.imag(H_temp)
        positions[i,:,:] = eve_pos
        distance_m[i,:] = dis_temp
        pathloss_db[i,:] = pl_temp
        distance_to_paired_lu[i,:] = eve_to_lu
        paired_lu_index[i,:] = pair_idx

        if (i+1)%2000==0:
            elapsed = time.time()-start_time
            print(f"Generated {i+1}/{number} strong-Eve samples, elapsed: {elapsed:.2f}s",flush=True)

    dataset_eve_true.data = torch.FloatTensor(data)
    dataset_eve_true.positions = torch.FloatTensor(positions)
    dataset_eve_true.distance_m = torch.FloatTensor(distance_m)
    dataset_eve_true.pathloss_db = torch.FloatTensor(pathloss_db)
    dataset_eve_true.distance_to_paired_lu = torch.FloatTensor(distance_to_paired_lu)
    dataset_eve_true.paired_lu_index = torch.LongTensor(paired_lu_index)

    print(f"Strong-Eve channel dataset generated, elapsed: {time.time()-start_time:.2f}s")

    if save_path:
        dataset_eve_true.save_dataset_to_filepath()
    return dataset_eve_true

def construct_estimated_dataset_from_true(dataset_true,mu,save_path):
    """Generate estimated Eve CSI under the relative-error uncertainty bound."""
    if mu < 0 or mu >= 1:
        raise ValueError("Closed-form construction requires 0 <= mu < 1")

    dataset_est = Generate_H_geometry_rician_dataset.__new__(Generate_H_geometry_rician_dataset)
    dataset_est.number = dataset_true.number
    dataset_est.K = dataset_true.K
    dataset_est.NT = dataset_true.NT
    dataset_est.d_min = dataset_true.d_min
    dataset_est.d_max = dataset_true.d_max
    dataset_est.rician_K_dB = dataset_true.rician_K_dB
    dataset_est.fc_hz = dataset_true.fc_hz
    dataset_est.scenario_name = dataset_true.scenario_name
    dataset_est.file_path = save_path
    dataset_est.mu = mu

    dataset_est.positions = dataset_true.positions.clone()
    dataset_est.distance_m = dataset_true.distance_m.clone()
    dataset_est.pathloss_db = dataset_true.pathloss_db.clone()

    if hasattr(dataset_true,'distance_to_paired_lu'):
        dataset_est.distance_to_paired_lu = dataset_true.distance_to_paired_lu.clone()
    if hasattr(dataset_true,'paired_lu_index'):
        dataset_est.paired_lu_index = dataset_true.paired_lu_index.clone()

    E_direction = torch.randn_like(dataset_true.data)
    norm_direction = torch.sqrt(
        torch.sum(E_direction**2,dim=[1,3],keepdim=True)
    ) + 1e-12
    E_direction = E_direction / norm_direction

    rho = torch.rand(
        dataset_true.data.shape[0],1,dataset_true.data.shape[2],1,
        dtype=dataset_true.data.dtype,
        device=dataset_true.data.device
    )
    alpha = mu * rho

    A = torch.sum(dataset_true.data**2,dim=[1,3],keepdim=True)
    c = torch.sum(dataset_true.data*E_direction,dim=[1,3],keepdim=True)

    alpha2 = alpha**2
    discriminant = alpha2**2*c**2 + alpha2*(1.0-alpha2)*A
    discriminant = torch.clamp(discriminant,min=0.0)

    radius = (
        -alpha2*c + torch.sqrt(discriminant)
    ) / (1.0-alpha2 + 1e-20)

    E_bounded = E_direction * radius
    dataset_est.data = dataset_true.data - E_bounded

    if save_path:
        dataset_est.save_dataset_to_filepath()
    return dataset_est

def verify_estimation_error_bound(dataset_true,dataset_est,mu):
    print("\n[Checking CSI error bound]")

    E = dataset_true.data-dataset_est.data
    norm_E = torch.sqrt(torch.sum(E**2,dim=[1,3]))
    norm_est = torch.sqrt(torch.sum(dataset_est.data**2,dim=[1,3]))+1e-20
    relative_error = norm_E/norm_est

    print(f"Maximum relative error mu: {mu:.8f}")
    print(f"Mean relative error: {relative_error.mean().item():.8f}")
    print(f"Minimum relative error: {relative_error.min().item():.8f}")
    print(f"Maximum relative error: {relative_error.max().item():.8f}")

    tolerance = 5e-5
    violation = relative_error > (mu+tolerance)
    violation_num = int(violation.sum().item())
    total_num = relative_error.numel()

    print(f"Constraint violations: {violation_num}/{total_num}")

    if violation_num == 0:
        print("Uncertainty-ball check: PASS")
    else:
        print("Uncertainty-ball check: WARNING")
    return relative_error

def verify_strong_eve_geometry(dataset_Eve):
    print("\n[Checking strong-Eve geometry]")

    if not hasattr(dataset_Eve,'distance_to_paired_lu'):
        print("Missing distance_to_paired_lu field")
        return

    d = dataset_Eve.distance_to_paired_lu

    print(f"Min Eve-to-paired-LU distance: {d.min().item():.3f} m")
    print(f"Max Eve-to-paired-LU distance: {d.max().item():.3f} m")
    print(f"Mean Eve-to-paired-LU distance: {d.mean().item():.3f} m")
    print(f"Mean Eve-to-BS distance: {dataset_Eve.distance_m.mean().item():.3f} m")
    print(f"Mean Eve path loss: {dataset_Eve.pathloss_db.mean().item():.3f} dB")

def verify_dataset_correlation(dataset_LU,dataset_Eve):
    """Report LU-Eve spatial correlations, including paired correlations."""
    print("\n[Checking spatial correlation]")

    H_LU_cp = torch.complex(dataset_LU.data[:,0],dataset_LU.data[:,1])
    H_Eve_cp = torch.complex(dataset_Eve.data[:,0],dataset_Eve.data[:,1])

    inner_prod = torch.matmul(H_LU_cp,H_Eve_cp.mH)
    norm_LU = torch.norm(H_LU_cp,dim=-1,keepdim=True)
    norm_Eve = torch.norm(H_Eve_cp,dim=-1,keepdim=True)
    norm_matrix = torch.matmul(norm_LU,norm_Eve.transpose(-1,-2))

    corr_matrix = torch.abs(inner_prod)/(norm_matrix+1e-20)
    avg_corr_matrix = corr_matrix.mean(dim=0).cpu()

    print("----- Mean spatial correlation (LU rows, Eve columns) -----")
    for k in range(dataset_LU.K):
        row_str = " ".join([f"{avg_corr_matrix[k,e].item():.4f}" for e in range(dataset_Eve.K)])
        print(f"LU_{k+1} correlations with Eves: [ {row_str} ]")
    print("---------------------------------------------")

    paired_number = min(dataset_LU.K,dataset_Eve.K)
    paired_corr = torch.stack([avg_corr_matrix[k,k] for k in range(paired_number)])
    print(f"Mean paired LU-Eve correlation: {paired_corr.mean().item():.4f}")
    return avg_corr_matrix

if __name__ =='__main__':

    number_train = 200000
    number_test = 10000

    with open('./config.yaml','r',encoding='utf-8') as f:
        config = yaml.safe_load(f)

    K = 4
    Eve = 4
    NT = 8

    mu = 0.1

    d_LU_min = 100.0
    d_LU_max = 500.0

    d_near_min = 20.0
    d_near_max = 50.0

    rician_K_dB = 10.0
    fc_hz = 3.5e9

    random_seed = 1000
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    root_path = './dataset_exp01_scenario'

    print("\n"+"="*70)
    print("[Geometry + Rician + Strong Eve]")
    print(f"K={K}, Eve={Eve}, NT={NT}, mu={mu}")
    print(f"LU distance range: [{d_LU_min},{d_LU_max}] m")
    print(f"Eve-to-paired-LU distance range: [{d_near_min},{d_near_max}] m")
    print(f"Rician K-factor           : {rician_K_dB} dB")
    print(f"Carrier frequency         : {fc_hz/1e9:.2f} GHz")
    print("="*70)

    lu_position_train_path = (
        f'{root_path}/dataset_position_K_train/'
        f'dataset_position_train_K={K}_dmin={d_LU_min}_dmax={d_LU_max}_number={number_train}.pt'
    )
    eve_position_train_path = (
        f'{root_path}/dataset_position_Eve_train/'
        f'dataset_position_train_Eve={Eve}_K={K}_dnearmin={d_near_min}_dnearmax={d_near_max}_number={number_train}.pt'
    )

    lu_position_test_path = (
        f'{root_path}/dataset_position_K_test/'
        f'dataset_position_test_K={K}_dmin={d_LU_min}_dmax={d_LU_max}_number={number_test}.pt'
    )
    eve_position_test_path = (
        f'{root_path}/dataset_position_Eve_test/'
        f'dataset_position_test_Eve={Eve}_K={K}_dnearmin={d_near_min}_dnearmax={d_near_max}_number={number_test}.pt'
    )

    print("\nLoading or creating fixed LU train positions...")
    positions_K_train,distance_K_train = load_or_create_lu_position_dataset(
        number_train,K,d_LU_min,d_LU_max,lu_position_train_path
    )

    print("\nLoading or creating fixed Eve train positions...")
    positions_Eve_train,distance_Eve_train,distance_Eve_to_LU_train,paired_index_train = load_or_create_eve_position_dataset(
        positions_K_train,Eve,d_near_min,d_near_max,eve_position_train_path
    )

    print("\nLoading or creating fixed LU test positions...")
    positions_K_test,distance_K_test = load_or_create_lu_position_dataset(
        number_test,K,d_LU_min,d_LU_max,lu_position_test_path
    )

    print("\nLoading or creating fixed Eve test positions...")
    positions_Eve_test,distance_Eve_test,distance_Eve_to_LU_test,paired_index_test = load_or_create_eve_position_dataset(
        positions_K_test,Eve,d_near_min,d_near_max,eve_position_test_path
    )

    print("\nCreating LU train channels...")
    dataset_K_train_path = f'{root_path}/dataset_K_train/dataset_train_K={K}_NT={NT}_KR={rician_K_dB}dB_number={number_train}.pt'
    dataset_K_train = load_dataset_if_compatible(
        dataset_K_train_path,K,NT,number_train,rician_K_dB,fc_hz,
        fixed_positions=positions_K_train
    )
    if dataset_K_train is None:
        dataset_K_train = Generate_H_geometry_rician_dataset(
            number_train,K,NT,d_LU_min,d_LU_max,
            rician_K_dB,fc_hz,
            dataset_K_train_path,
            fixed_positions=positions_K_train
        )

    print("\nCreating true Eve train channels...")
    dataset_Eve_train_true_path = f'{root_path}/dataset_Eve_train_true/dataset_train_true_Eve={Eve}_NT={NT}_KR={rician_K_dB}dB_number={number_train}.pt'
    dataset_Eve_train_true = load_dataset_if_compatible(
        dataset_Eve_train_true_path,Eve,NT,number_train,rician_K_dB,fc_hz,
        fixed_positions=positions_Eve_train
    )
    if dataset_Eve_train_true is None:
        dataset_Eve_train_true = construct_strong_eve_true_dataset(
            dataset_K_train,Eve,d_near_min,d_near_max,
            rician_K_dB,fc_hz,
            dataset_Eve_train_true_path,
            fixed_eve_positions=positions_Eve_train,
            fixed_distance_to_paired_lu=distance_Eve_to_LU_train,
            fixed_paired_lu_index=paired_index_train
        )

    print("\nCreating estimated Eve train channels...")
    dataset_Eve_train_est_path = f'{root_path}/dataset_Eve_train_est/dataset_train_est_Eve={Eve}_NT={NT}_KR={rician_K_dB}dB_mu={mu}_number={number_train}.pt'
    dataset_Eve_train_est = load_dataset_if_compatible(
        dataset_Eve_train_est_path,Eve,NT,number_train,rician_K_dB,fc_hz,
        fixed_positions=positions_Eve_train,mu=mu
    )
    if dataset_Eve_train_est is None:
        dataset_Eve_train_est = construct_estimated_dataset_from_true(
            dataset_Eve_train_true,mu,
            dataset_Eve_train_est_path
        )

    print("\nCreating LU test channels...")
    dataset_K_test_path = f'{root_path}/dataset_K_test/dataset_test_K={K}_NT={NT}_KR={rician_K_dB}dB_number={number_test}.pt'
    dataset_K_test = load_dataset_if_compatible(
        dataset_K_test_path,K,NT,number_test,rician_K_dB,fc_hz,
        fixed_positions=positions_K_test
    )
    if dataset_K_test is None:
        dataset_K_test = Generate_H_geometry_rician_dataset(
            number_test,K,NT,d_LU_min,d_LU_max,
            rician_K_dB,fc_hz,
            dataset_K_test_path,
            fixed_positions=positions_K_test
        )

    print("\nCreating true Eve test channels...")
    dataset_Eve_test_true_path = f'{root_path}/dataset_Eve_test_true/dataset_test_true_Eve={Eve}_NT={NT}_KR={rician_K_dB}dB_number={number_test}.pt'
    dataset_Eve_test_true = load_dataset_if_compatible(
        dataset_Eve_test_true_path,Eve,NT,number_test,rician_K_dB,fc_hz,
        fixed_positions=positions_Eve_test
    )
    if dataset_Eve_test_true is None:
        dataset_Eve_test_true = construct_strong_eve_true_dataset(
            dataset_K_test,Eve,d_near_min,d_near_max,
            rician_K_dB,fc_hz,
            dataset_Eve_test_true_path,
            fixed_eve_positions=positions_Eve_test,
            fixed_distance_to_paired_lu=distance_Eve_to_LU_test,
            fixed_paired_lu_index=paired_index_test
        )

    print("\nCreating estimated Eve test channels...")
    dataset_Eve_test_est_path = f'{root_path}/dataset_Eve_test_est/dataset_test_est_Eve={Eve}_NT={NT}_KR={rician_K_dB}dB_mu={mu}_number={number_test}.pt'
    dataset_Eve_test_est = load_dataset_if_compatible(
        dataset_Eve_test_est_path,Eve,NT,number_test,rician_K_dB,fc_hz,
        fixed_positions=positions_Eve_test,mu=mu
    )
    if dataset_Eve_test_est is None:
        dataset_Eve_test_est = construct_estimated_dataset_from_true(
            dataset_Eve_test_true,mu,
            dataset_Eve_test_est_path
        )

    verify_strong_eve_geometry(dataset_Eve_test_true)
    verify_estimation_error_bound(dataset_Eve_test_true,dataset_Eve_test_est,mu)
    verify_dataset_correlation(dataset_K_test,dataset_Eve_test_true)

    print("\n"+"="*70)
    print("[Dataset flow summary]")
    print(f"Train: LU perfect CSI samples: {len(dataset_K_train)}")
    print(f"Train: estimated Eve CSI samples: {len(dataset_Eve_train_est)}")
    print(f"Inference: estimated Eve CSI samples: {len(dataset_Eve_test_est)}")
    print(f"Evaluation: true Eve CSI samples: {len(dataset_Eve_test_true)}")
    print("Worst-case evaluation uses online PGD around estimated Eve CSI.")
    print("="*70)
