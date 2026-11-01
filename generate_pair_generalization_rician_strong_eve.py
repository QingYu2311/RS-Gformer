import torch
from torch.utils.data import Dataset
import numpy as np
import time,os

from generate_geometry_rician_strong_eve import generate_positions_in_annulus,\
                                                  generate_eve_positions_near_lu,\
                                                  generate_H_rician_from_positions,\
                                                  Generate_H_geometry_rician_dataset,\
                                                  construct_strong_eve_true_dataset,\
                                                  construct_estimated_dataset_from_true,\
                                                  load_or_create_lu_position_dataset,\
                                                  load_or_create_eve_position_dataset,\
                                                  load_dataset_if_compatible,\
                                                  verify_strong_eve_geometry,\
                                                  verify_estimation_error_bound,\
                                                  verify_dataset_correlation

ERROR_SAMPLING_MODE = 'inside'

def _sample_relative_error_level(reference_tensor, mu, mode):
    """Draw relative CSI errors inside the ball or on its surface."""
    if mode == 'surface':
        return torch.full_like(reference_tensor, float(mu))
    elif mode == 'inside':
        return torch.rand_like(reference_tensor) * float(mu)
    else:
        raise ValueError("ERROR_SAMPLING_MODE must be 'inside' or 'surface'")

def _solve_relative_error_radius(H_true, E_direction, reduce_dims, alpha):
    """Solve the radius ensuring ||E|| = alpha * ||H_est||."""
    A = torch.sum(H_true**2, dim=reduce_dims, keepdim=True)
    c = torch.sum(H_true*E_direction, dim=reduce_dims, keepdim=True)
    alpha2 = alpha**2
    discriminant = alpha2**2*c**2 + alpha2*(1.0-alpha2)*A
    discriminant = torch.clamp(discriminant, min=0.0)
    radius = (-alpha2*c + torch.sqrt(discriminant)) / (1.0-alpha2 + 1e-20)
    return radius

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
    norm_direction = torch.sqrt(torch.sum(E_direction**2,dim=[1,3],keepdim=True))+1e-12
    E_direction = E_direction/norm_direction

    A_ref = torch.sum(dataset_true.data**2,dim=[1,3],keepdim=True)
    alpha = _sample_relative_error_level(A_ref,mu,ERROR_SAMPLING_MODE)
    radius = _solve_relative_error_radius(dataset_true.data,E_direction,[1,3],alpha)

    E_bounded = E_direction*radius
    dataset_est.data = dataset_true.data-E_bounded

    if save_path:
        dataset_est.save_dataset_to_filepath()
    return dataset_est

def verify_estimation_error_bound(dataset_true,dataset_est,mu):
    print("\n[Checking CSI error bound]")
    E = dataset_true.data-dataset_est.data
    norm_E = torch.sqrt(torch.sum(E**2,dim=[1,3]))
    norm_est = torch.sqrt(torch.sum(dataset_est.data**2,dim=[1,3]))+1e-20
    relative_error = norm_E/norm_est

    print(f"Sampling mode: {ERROR_SAMPLING_MODE}")
    print(f"Maximum target relative error mu: {mu:.8f}")
    print(f"Mean relative error: {relative_error.mean().item():.8f}")
    print(f"Minimum relative error: {relative_error.min().item():.8f}")
    print(f"Maximum relative error: {relative_error.max().item():.8f}")

    tolerance = 5e-5
    if ERROR_SAMPLING_MODE == 'surface':
        max_abs_deviation = torch.max(torch.abs(relative_error-mu)).item()
        print(f"Maximum absolute deviation: {max_abs_deviation:.8e}")
        print("Uncertainty-sphere check: PASS" if max_abs_deviation<tolerance else "Uncertainty-sphere check: WARNING")
    else:
        violation_num = int((relative_error>(mu+tolerance)).sum().item())
        total_num = relative_error.numel()
        print(f"Constraint violations: {violation_num}/{total_num}")
        print("Uncertainty-ball check: PASS" if violation_num==0 else "Uncertainty-ball check: WARNING")
    return relative_error

def generate_truncated_pair_array(number,exp_mean,max_pair_train,random_seed):
    """Draw exponentially distributed pair counts and clip to max_pair_train."""
    rng = np.random.default_rng(random_seed)
    pair_array = np.round(rng.exponential(scale=exp_mean,size=number)).astype(int)+1
    pair_array = np.clip(pair_array,1,max_pair_train)
    return pair_array

def load_or_create_pair_number_array(number,exp_mean,max_pair_train,file_path,random_seed):
    if os.path.exists(file_path):
        saved_data = torch.load(file_path,map_location='cpu')
        pair_array = saved_data['pair_array'].cpu().numpy().astype(np.int64)

        if saved_data['number'] != number:
            raise ValueError(f"Existing pair-count dataset number mismatch: {saved_data['number']} != {number}")
        if abs(float(saved_data['exp_mean'])-float(exp_mean))>1e-12:
            raise ValueError(f"Existing pair-count dataset exp_mean mismatch: {saved_data['exp_mean']} != {exp_mean}")
        if saved_data['max_pair_train'] != max_pair_train:
            raise ValueError(f"Existing pair-count dataset max_pair_train mismatch: "
                             f"{saved_data['max_pair_train']} != {max_pair_train}")

        print(f"Reusing existing training pair counts: {file_path}")
        return pair_array

    pair_array = generate_truncated_pair_array(
        number,exp_mean,max_pair_train,random_seed)

    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    torch.save({
        'pair_array':torch.LongTensor(pair_array),
        'number':number,
        'exp_mean':exp_mean,
        'max_pair_train':max_pair_train,
        'random_seed':random_seed
    },file_path)

    print(f"Training pair counts saved to: {file_path}")
    return pair_array

def print_pair_number_distribution(pair_array,max_pair_train):
    print("\n"+"="*70)
    print("[Training LU-Eve pair distribution]")

    total_number = len(pair_array)
    for pair_number in range(1,max_pair_train+1):
        count = int(np.sum(pair_array==pair_number))
        percentage = count/total_number*100.0
        print(f"K=Eve={pair_number:2d}: {count:6d} samples, {percentage:7.3f}%")

    print("-"*70)
    print(f"Total samples: {total_number}")
    print(f"Mean pair count: {pair_array.mean():.4f}")
    print(f"Minimum pair count: {pair_array.min()}")
    print(f"Maximum pair count: {pair_array.max()}")
    print("="*70)

def load_or_create_variable_lu_position_dataset(pair_array,d_min,d_max,file_path):
    number = len(pair_array)

    if os.path.exists(file_path):
        saved_data = torch.load(file_path,map_location='cpu')
        saved_pair_array = saved_data['pair_array'].cpu().numpy().astype(np.int64)

        if saved_data['number'] != number:
            raise ValueError(f"Existing variable-LU positions number mismatch: "
                             f"{saved_data['number']} != {number}")
        if not np.array_equal(saved_pair_array,pair_array):
            raise ValueError(f"Existing variable-LU positions pair_array mismatch: {file_path}")

        print(f"Reusing variable-LU positions: {file_path}")
        return saved_data['positions'],saved_data['distance_m']

    start_time = time.time()
    positions_list,distance_list = [],[]

    for i in range(number):
        K_i = int(pair_array[i])
        positions,distance_m = generate_positions_in_annulus(K_i,d_min,d_max)

        positions_list.append(torch.FloatTensor(positions))
        distance_list.append(torch.FloatTensor(distance_m))

        if (i+1)%2000==0:
            elapsed = time.time()-start_time
            print(f"Generated {i+1}/{number} variable-LU position samples, elapsed: {elapsed:.2f}s",flush=True)

    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    torch.save({
        'positions':positions_list,
        'distance_m':distance_list,
        'pair_array':torch.LongTensor(pair_array),
        'number':number,
        'd_min':d_min,
        'd_max':d_max,
        'scenario_name':'pair_generalization_geo_rician_strong_eve'
    },file_path)

    print(f"Variable-LU positions saved to: {file_path}")
    return positions_list,distance_list

def load_or_create_variable_eve_position_dataset(
        lu_positions_list,pair_array,d_near_min,d_near_max,file_path):

    number = len(pair_array)

    if os.path.exists(file_path):
        saved_data = torch.load(file_path,map_location='cpu')
        saved_pair_array = saved_data['pair_array'].cpu().numpy().astype(np.int64)

        if saved_data['number'] != number:
            raise ValueError(f"Existing variable-Eve positions number mismatch: "
                             f"{saved_data['number']} != {number}")
        if not np.array_equal(saved_pair_array,pair_array):
            raise ValueError(f"Existing variable-Eve positions pair_array mismatch: {file_path}")

        print(f"Reusing variable-Eve positions: {file_path}")
        return saved_data['positions'],saved_data['distance_m'],\
               saved_data['distance_to_paired_lu'],saved_data['paired_lu_index']

    start_time = time.time()
    positions_list,distance_list = [],[]
    distance_to_lu_list,paired_index_list = [],[]

    for i in range(number):
        Eve_i = int(pair_array[i])
        lu_positions = lu_positions_list[i].cpu().numpy()

        eve_positions,eve_distance,eve_to_lu,paired_index = generate_eve_positions_near_lu(
            lu_positions,Eve_i,d_near_min,d_near_max)

        positions_list.append(torch.FloatTensor(eve_positions))
        distance_list.append(torch.FloatTensor(eve_distance))
        distance_to_lu_list.append(torch.FloatTensor(eve_to_lu))
        paired_index_list.append(torch.LongTensor(paired_index))

        if (i+1)%2000==0:
            elapsed = time.time()-start_time
            print(f"Generated {i+1}/{number} variable-Eve position samples, "
                  f"elapsed: {elapsed:.2f}s",flush=True)

    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    torch.save({
        'positions':positions_list,
        'distance_m':distance_list,
        'distance_to_paired_lu':distance_to_lu_list,
        'paired_lu_index':paired_index_list,
        'pair_array':torch.LongTensor(pair_array),
        'number':number,
        'd_near_min':d_near_min,
        'd_near_max':d_near_max,
        'scenario_name':'pair_generalization_geo_rician_strong_eve'
    },file_path)

    print(f"Variable-Eve positions saved to: {file_path}")
    return positions_list,distance_list,distance_to_lu_list,paired_index_list

class Generate_H_geometry_rician_variable_pair_dataset(Dataset):
    def __init__(
            self,pair_array,NT,rician_K_dB,fc_hz,positions_list,
            file_path=None,exp_mean=None,max_pair_train=None,
            distance_to_paired_lu=None,paired_lu_index=None):

        super().__init__()
        start_time = time.time()

        self.number = len(pair_array)
        self.pair_array = torch.LongTensor(pair_array)
        self.exp_mean = exp_mean
        self.max_pair_train = max_pair_train
        self.NT = NT
        self.rician_K_dB = rician_K_dB
        self.fc_hz = fc_hz
        self.file_path = file_path
        self.scenario_name = 'pair_generalization_geo_rician_strong_eve'

        self.data = []
        self.positions = []
        self.distance_m = []
        self.pathloss_db = []

        for i in range(self.number):
            pair_number = int(pair_array[i])

            if torch.is_tensor(positions_list[i]):
                positions = positions_list[i].cpu().numpy()
            else:
                positions = positions_list[i]

            if positions.shape != (pair_number,2):
                raise ValueError(f"Position sample {i} has an invalid shape, "
                                 f"expected {(pair_number,2)} , got {positions.shape}")

            H_temp,distance_m,pathloss_db = generate_H_rician_from_positions(
                positions,NT,rician_K_dB,fc_hz=fc_hz)

            H_ri = np.stack([np.real(H_temp),np.imag(H_temp)],axis=0).astype(np.float32)

            self.data.append(torch.FloatTensor(H_ri))
            self.positions.append(torch.FloatTensor(positions))
            self.distance_m.append(torch.FloatTensor(distance_m))
            self.pathloss_db.append(torch.FloatTensor(pathloss_db))

            if (i+1)%2000==0:
                elapsed = time.time()-start_time
                print(f"Generated {i+1}/{self.number} variable-pair Rician channels, "
                      f"elapsed: {elapsed:.2f}s",flush=True)

        if distance_to_paired_lu is not None:
            self.distance_to_paired_lu = [x.clone() for x in distance_to_paired_lu]

        if paired_lu_index is not None:
            self.paired_lu_index = [x.clone() for x in paired_lu_index]

        print(f"Variable-pair Geometry + Rician dataset generated, "
              f"elapsed: {time.time()-start_time:.2f}s")

        if file_path:
            self.save_dataset_to_filepath()

    def save_dataset_to_filepath(self):
        os.makedirs(os.path.dirname(self.file_path),exist_ok=True)

        save_dict = {
            'data':self.data,
            'positions':self.positions,
            'distance_m':self.distance_m,
            'pathloss_db':self.pathloss_db,
            'pair_array':self.pair_array,
            'exp_mean':self.exp_mean,
            'max_pair_train':self.max_pair_train,
            'NT':self.NT,
            'number':self.number,
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

def load_variable_pair_dataset(filename):
    saved_data = torch.load(filename,map_location='cpu')

    dataset = Generate_H_geometry_rician_variable_pair_dataset.__new__(
        Generate_H_geometry_rician_variable_pair_dataset)

    dataset.data = saved_data['data']
    dataset.positions = saved_data['positions']
    dataset.distance_m = saved_data['distance_m']
    dataset.pathloss_db = saved_data['pathloss_db']
    dataset.pair_array = saved_data['pair_array']
    dataset.exp_mean = saved_data.get('exp_mean',None)
    dataset.max_pair_train = saved_data.get('max_pair_train',None)
    dataset.NT = saved_data['NT']
    dataset.number = saved_data['number']
    dataset.rician_K_dB = saved_data['rician_K_dB']
    dataset.fc_hz = saved_data['fc_hz']
    dataset.scenario_name = saved_data.get(
        'scenario_name','pair_generalization_geo_rician_strong_eve')
    dataset.file_path = filename

    if 'mu' in saved_data:
        dataset.mu = saved_data['mu']
    if 'distance_to_paired_lu' in saved_data:
        dataset.distance_to_paired_lu = saved_data['distance_to_paired_lu']
    if 'paired_lu_index' in saved_data:
        dataset.paired_lu_index = saved_data['paired_lu_index']

    print(f"Dataset loaded from {filename}")
    print(f"Dataset: mean pair count={dataset.pair_array.float().mean().item():.4f}, "
          f"NT={dataset.NT}, samples={dataset.number}")

    return dataset

def load_variable_pair_dataset_if_compatible(
        filename,pair_array,NT,rician_K_dB,fc_hz,
        exp_mean,max_pair_train,mu=None):

    if not os.path.exists(filename):
        return None

    dataset = load_variable_pair_dataset(filename)
    pair_array_tensor = torch.LongTensor(pair_array)

    is_compatible = True
    is_compatible = is_compatible and dataset.number == len(pair_array)
    is_compatible = is_compatible and dataset.NT == NT
    is_compatible = is_compatible and torch.equal(dataset.pair_array,pair_array_tensor)
    is_compatible = is_compatible and abs(float(dataset.rician_K_dB)-float(rician_K_dB))<1e-12
    is_compatible = is_compatible and abs(float(dataset.fc_hz)-float(fc_hz))<1e-6
    is_compatible = is_compatible and dataset.exp_mean is not None

    if dataset.exp_mean is not None:
        is_compatible = is_compatible and abs(float(dataset.exp_mean)-float(exp_mean))<1e-12

    is_compatible = is_compatible and dataset.max_pair_train == max_pair_train

    if mu is not None:
        is_compatible = is_compatible and hasattr(dataset,'mu')
        if hasattr(dataset,'mu'):
            is_compatible = is_compatible and abs(float(dataset.mu)-float(mu))<1e-12

    if is_compatible:
        print(f"Reusing compatible variable-pair channels: {filename}")
        return dataset

    print(f"Regenerating incompatible variable-pair channels: {filename}")
    return None

def construct_estimated_dataset_from_true_variable_pair(dataset_true,mu,save_path):
    """Construct estimated Eve CSI with per-pair relative error bounds."""
    if mu < 0 or mu >= 1:
        raise ValueError("Closed-form construction requires 0 <= mu < 1")

    dataset_est = Generate_H_geometry_rician_variable_pair_dataset.__new__(
        Generate_H_geometry_rician_variable_pair_dataset)

    dataset_est.number = dataset_true.number
    dataset_est.pair_array = dataset_true.pair_array.clone()
    dataset_est.exp_mean = dataset_true.exp_mean
    dataset_est.max_pair_train = dataset_true.max_pair_train
    dataset_est.NT = dataset_true.NT
    dataset_est.rician_K_dB = dataset_true.rician_K_dB
    dataset_est.fc_hz = dataset_true.fc_hz
    dataset_est.file_path = save_path
    dataset_est.scenario_name = dataset_true.scenario_name
    dataset_est.mu = mu

    dataset_est.positions = [x.clone() for x in dataset_true.positions]
    dataset_est.distance_m = [x.clone() for x in dataset_true.distance_m]
    dataset_est.pathloss_db = [x.clone() for x in dataset_true.pathloss_db]

    if hasattr(dataset_true,'distance_to_paired_lu'):
        dataset_est.distance_to_paired_lu = [x.clone() for x in dataset_true.distance_to_paired_lu]
    if hasattr(dataset_true,'paired_lu_index'):
        dataset_est.paired_lu_index = [x.clone() for x in dataset_true.paired_lu_index]

    dataset_est.data = []

    for H_true in dataset_true.data:

        E_direction = torch.randn_like(H_true)
        norm_direction = torch.sqrt(torch.sum(E_direction**2,dim=[0,2],keepdim=True))+1e-12
        E_direction = E_direction/norm_direction

        A_ref = torch.sum(H_true**2,dim=[0,2],keepdim=True)
        alpha = _sample_relative_error_level(A_ref,mu,ERROR_SAMPLING_MODE)
        radius = _solve_relative_error_radius(H_true,E_direction,[0,2],alpha)

        E_bounded = E_direction*radius
        H_est = H_true-E_bounded
        dataset_est.data.append(H_est)

    if save_path:
        dataset_est.save_dataset_to_filepath()
    return dataset_est

def verify_estimation_error_bound_variable_pair(dataset_true,dataset_est,mu):
    print("\n[Checking variable-pair CSI error bound]")

    relative_error_list = []
    for H_true,H_est in zip(dataset_true.data,dataset_est.data):
        E = H_true-H_est
        norm_E = torch.sqrt(torch.sum(E**2,dim=[0,2]))
        norm_est = torch.sqrt(torch.sum(H_est**2,dim=[0,2]))+1e-20
        relative_error_list.append(norm_E/norm_est)

    relative_error = torch.cat(relative_error_list)

    print(f"Sampling mode: {ERROR_SAMPLING_MODE}")
    print(f"Maximum target relative error mu: {mu:.8f}")
    print(f"Mean relative error: {relative_error.mean().item():.8f}")
    print(f"Minimum relative error: {relative_error.min().item():.8f}")
    print(f"Maximum relative error: {relative_error.max().item():.8f}")

    tolerance = 5e-5
    if ERROR_SAMPLING_MODE == 'surface':
        max_abs_deviation = torch.max(torch.abs(relative_error-mu)).item()
        print(f"Maximum absolute deviation: {max_abs_deviation:.8e}")
        print("Uncertainty-sphere check: PASS" if max_abs_deviation<tolerance else "Uncertainty-sphere check: WARNING")
    else:
        violation_num = int((relative_error>(mu+tolerance)).sum().item())
        total_num = relative_error.numel()
        print(f"Constraint violations: {violation_num}/{total_num}")
        print("Uncertainty-ball check: PASS" if violation_num==0 else "Uncertainty-ball check: WARNING")

    return relative_error

def verify_variable_pair_geometry_and_correlation(dataset_LU,dataset_Eve):
    print("\n[Checking variable-pair strong-Eve geometry and correlation]")

    distance_all = torch.cat(dataset_Eve.distance_to_paired_lu)
    eve_bs_distance_all = torch.cat(dataset_Eve.distance_m)

    print(f"Min Eve-to-paired-LU distance: {distance_all.min().item():.3f} m")
    print(f"Max Eve-to-paired-LU distance: {distance_all.max().item():.3f} m")
    print(f"Mean Eve-to-paired-LU distance: {distance_all.mean().item():.3f} m")
    print(f"Mean Eve-to-BS distance: {eve_bs_distance_all.mean().item():.3f} m")

    paired_corr_list = []

    for H_LU_ri,H_Eve_ri,pair_idx in zip(
            dataset_LU.data,dataset_Eve.data,dataset_Eve.paired_lu_index):

        H_LU = torch.complex(H_LU_ri[0],H_LU_ri[1])
        H_Eve = torch.complex(H_Eve_ri[0],H_Eve_ri[1])

        for j,k in enumerate(pair_idx):
            h = H_LU[int(k)]
            g = H_Eve[j]

            corr = torch.abs(torch.sum(h*torch.conj(g))) / (
                torch.norm(h)*torch.norm(g)+1e-20)

            paired_corr_list.append(corr)

    paired_corr = torch.stack(paired_corr_list)

    print(f"Mean paired LU-Eve correlation: {paired_corr.mean().item():.4f}")
    print(f"Minimum paired correlation: {paired_corr.min().item():.4f}")
    print(f"Maximum paired correlation: {paired_corr.max().item():.4f}")

if __name__ =='__main__':

    number_train = 500000
    number_test = 10000

    Kmean = Evemean = exp_mean = 4
    max_pair_train = 10

    pair_test_list = [2,4,6,8,10,12,14,16]

    NT = 16
    mu_train = 0.1
    mu_test = 0.1

    d_LU_min = 100.0
    d_LU_max = 500.0

    d_near_min = 20.0
    d_near_max = 50.0

    rician_K_dB = 10.0
    fc_hz = 3.5e9

    random_seed = 2026
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    root_path = './dataset_pair_generalization_rician_strong_eve'

    print("\n"+"="*80)
    print("[Variable-pair Geometry + Rician + Strong Eve]")
    print(f"Train samples={number_train}, test samples={number_test}")
    print(f"Exponential mean pair count={exp_mean}, upper bound={max_pair_train}")
    print(f"Fixed test pair counts={pair_test_list}")
    print(f"NT={NT}, mu_train={mu_train}, mu_test={mu_test}")
    print(f"LU distance range=[{d_LU_min},{d_LU_max}] m")
    print(f"Eve-to-paired-LU distance range=[{d_near_min},{d_near_max}] m")
    print(f"Rician K-factor={rician_K_dB} dB, fc={fc_hz/1e9:.2f} GHz")
    print("="*80)

    pair_array_path = (
        f'{root_path}/dataset_pair_number_train/'
        f'pair_number_train_mean={exp_mean}_max={max_pair_train}_number={number_train}.pt'
    )

    pair_array = load_or_create_pair_number_array(
        number_train,exp_mean,max_pair_train,pair_array_path,random_seed)

    print_pair_number_distribution(pair_array,max_pair_train)

    lu_position_train_path = (
        f'{root_path}/dataset_position_K_train/'
        f'dataset_position_train_Kmean={Kmean}_Kmax={max_pair_train}_'
        f'dmin={d_LU_min}_dmax={d_LU_max}_number={number_train}.pt'
    )

    eve_position_train_path = (
        f'{root_path}/dataset_position_Eve_train/'
        f'dataset_position_train_Evemean={Evemean}_Evemax={max_pair_train}_'
        f'dnearmin={d_near_min}_dnearmax={d_near_max}_number={number_train}.pt'
    )

    print("\nLoading/creating variable-LU train positions...")
    positions_K_train,distance_K_train = load_or_create_variable_lu_position_dataset(
        pair_array,d_LU_min,d_LU_max,lu_position_train_path)

    print("\nLoading/creating variable-Eve train positions...")
    positions_Eve_train,distance_Eve_train,distance_Eve_to_LU_train,paired_index_train = \
        load_or_create_variable_eve_position_dataset(
            positions_K_train,pair_array,d_near_min,d_near_max,eve_position_train_path)

    print("\nCreating LU train channels...")
    dataset_K_train_path = (
        f'{root_path}/dataset_K_train/'
        f'dataset_train_Kmean={Kmean}_Kmax={max_pair_train}_NT={NT}_'
        f'KR={rician_K_dB}dB_number={number_train}.pt'
    )

    dataset_K_train = load_variable_pair_dataset_if_compatible(
        dataset_K_train_path,pair_array,NT,rician_K_dB,fc_hz,
        exp_mean,max_pair_train)

    if dataset_K_train is None:
        dataset_K_train = Generate_H_geometry_rician_variable_pair_dataset(
            pair_array,NT,rician_K_dB,fc_hz,positions_K_train,
            dataset_K_train_path,exp_mean,max_pair_train)

    print("\nCreating true Eve train channels...")
    dataset_Eve_train_true_path = (
        f'{root_path}/dataset_Eve_train_true/'
        f'dataset_train_true_Evemean={Evemean}_Evemax={max_pair_train}_NT={NT}_'
        f'KR={rician_K_dB}dB_number={number_train}.pt'
    )

    dataset_Eve_train_true = load_variable_pair_dataset_if_compatible(
        dataset_Eve_train_true_path,pair_array,NT,rician_K_dB,fc_hz,
        exp_mean,max_pair_train)

    if dataset_Eve_train_true is None:
        dataset_Eve_train_true = Generate_H_geometry_rician_variable_pair_dataset(
            pair_array,NT,rician_K_dB,fc_hz,positions_Eve_train,
            dataset_Eve_train_true_path,exp_mean,max_pair_train,
            distance_to_paired_lu=distance_Eve_to_LU_train,
            paired_lu_index=paired_index_train)

    print("\nCreating estimated Eve train channels...")
    dataset_Eve_train_est_path = (
        f'{root_path}/dataset_Eve_train_est/'
        f'dataset_train_est_Evemean={Evemean}_Evemax={max_pair_train}_NT={NT}_'
        f'KR={rician_K_dB}dB_mu={mu_train}_number={number_train}.pt'
    )

    dataset_Eve_train_est = load_variable_pair_dataset_if_compatible(
        dataset_Eve_train_est_path,pair_array,NT,rician_K_dB,fc_hz,
        exp_mean,max_pair_train,mu=mu_train)

    if dataset_Eve_train_est is None:
        dataset_Eve_train_est = construct_estimated_dataset_from_true_variable_pair(
            dataset_Eve_train_true,mu_train,dataset_Eve_train_est_path)

    print("\n"+"="*80)
    print("Generating fixed-pair test datasets")
    print("="*80)

    for pair_number in pair_test_list:
        K = Eve = pair_number

        print("\n"+"-"*70)
        print(f"Generating test set: K={K}, Eve={Eve}, NT={NT}")

        if pair_number>max_pair_train:
            print(f"Current K=Eve={pair_number} > maximum training pair count={max_pair_train}; OOD size")

        print("-"*70)

        lu_position_test_path = (
            f'{root_path}/dataset_position_K_test/'
            f'dataset_position_test_K={K}_dmin={d_LU_min}_dmax={d_LU_max}_'
            f'number={number_test}.pt'
        )

        positions_K_test,distance_K_test = load_or_create_lu_position_dataset(
            number_test,K,d_LU_min,d_LU_max,lu_position_test_path)

        eve_position_test_path = (
            f'{root_path}/dataset_position_Eve_test/'
            f'dataset_position_test_Eve={Eve}_K={K}_dnearmin={d_near_min}_'
            f'dnearmax={d_near_max}_number={number_test}.pt'
        )

        positions_Eve_test,distance_Eve_test,distance_Eve_to_LU_test,paired_index_test = \
            load_or_create_eve_position_dataset(
                positions_K_test,Eve,d_near_min,d_near_max,eve_position_test_path)

        print("\nCreating LU test channels...")
        dataset_K_test_path = (
            f'{root_path}/dataset_K_test/'
            f'dataset_test_K={K}_NT={NT}_KR={rician_K_dB}dB_number={number_test}.pt'
        )

        dataset_K_test = load_dataset_if_compatible(
            dataset_K_test_path,K,NT,number_test,rician_K_dB,fc_hz,
            fixed_positions=positions_K_test)

        if dataset_K_test is None:
            dataset_K_test = Generate_H_geometry_rician_dataset(
                number_test,K,NT,d_LU_min,d_LU_max,
                rician_K_dB,fc_hz,dataset_K_test_path,
                fixed_positions=positions_K_test)

        print("\nCreating true Eve test channels...")
        dataset_Eve_test_true_path = (
            f'{root_path}/dataset_Eve_test_true/'
            f'dataset_test_true_Eve={Eve}_K={K}_NT={NT}_'
            f'KR={rician_K_dB}dB_number={number_test}.pt'
        )

        dataset_Eve_test_true = load_dataset_if_compatible(
            dataset_Eve_test_true_path,Eve,NT,number_test,rician_K_dB,fc_hz,
            fixed_positions=positions_Eve_test)

        if dataset_Eve_test_true is None:
            dataset_Eve_test_true = construct_strong_eve_true_dataset(
                dataset_K_test,Eve,d_near_min,d_near_max,
                rician_K_dB,fc_hz,dataset_Eve_test_true_path,
                fixed_eve_positions=positions_Eve_test,
                fixed_distance_to_paired_lu=distance_Eve_to_LU_test,
                fixed_paired_lu_index=paired_index_test)

        print("\nCreating estimated Eve test channels...")
        dataset_Eve_test_est_path = (
            f'{root_path}/dataset_Eve_test_est/'
            f'dataset_test_est_Eve={Eve}_K={K}_NT={NT}_KR={rician_K_dB}dB_'
            f'mu={mu_test}_number={number_test}.pt'
        )

        dataset_Eve_test_est = load_dataset_if_compatible(
            dataset_Eve_test_est_path,Eve,NT,number_test,rician_K_dB,fc_hz,
            fixed_positions=positions_Eve_test,mu=mu_test)

        if dataset_Eve_test_est is None:
            dataset_Eve_test_est = construct_estimated_dataset_from_true(
                dataset_Eve_test_true,mu_test,dataset_Eve_test_est_path)

        verify_strong_eve_geometry(dataset_Eve_test_true)
        verify_estimation_error_bound(dataset_Eve_test_true,dataset_Eve_test_est,mu_test)
        verify_dataset_correlation(dataset_K_test,dataset_Eve_test_true)

    verify_variable_pair_geometry_and_correlation(
        dataset_K_train,dataset_Eve_train_true)

    verify_estimation_error_bound_variable_pair(
        dataset_Eve_train_true,dataset_Eve_train_est,mu_train)

    print("\n"+"="*80)
    print("[Variable-pair dataset generation complete]")
    print(f"Train: exponential K_i=Eve_i, mean={exp_mean}, truncated at {max_pair_train}")
    print(f"Train: perfect LU CSI samples: {len(dataset_K_train)}")
    print(f"Train: estimated Eve CSI samples: {len(dataset_Eve_train_est)}")
    print(f"Fixed test pair counts:{pair_test_list}")
    print(f"Samples per fixed-pair test set: {number_test}")
    print("Inference: true LU CSI + estimated Eve CSI")
    print("Actual-channel evaluation: true Eve CSI")
    print("Worst-case evaluation: online PGD around estimated Eve CSI")
    print("="*80)
