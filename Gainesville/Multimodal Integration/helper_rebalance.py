from env_v12_13_rebalance import *
import tensorflow as tf
from scipy.spatial import cKDTree
# from libpysal.weights import KNN
# from esda import Moran
# from spatial_statistics import global_morans_I
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import geopandas as gpd
from scipy.stats import entropy
import ot
from scipy.special import rel_entr

def create_study_area(services, max_scooter_per_area, deploy_cost, docked_cost, N, no_scooter_penalty, pickup_reward, equity_penalty, \
                      rider_test_list, logistics_car_cost_per_mile, logistics_car_capacity, penalty_dist_bus_stop, reward_num_bus_stop):
    """Creates an object of the Service class from the env module.

    Args:
       services (list): List of mm-tuples named "index", 'latitude', 'longitude', 'departure_rate', 'arrival_rate',
       'dist_transPOI', 'num_transPOI', 'dist_bus_stop', 'num_bus_stop', 'geometry'.
       max_scooter_per_area (int): Max number of scooters each service area contains.
       scooter_no: number of initital scooters at each service area
       deploy_cost (float): Cost of deploying a scooter.
       docked_cost (float): Cost of deploying a docked station
       N: Number of service areas.
       no_scooter_penalty (float): Penalty for deploying a scooter when there is none.
       pickup_reward (float): Reward for serving a riders.

    Returns:
       Study area created.

    """
    # Creates a study area with N service areas.
    study_area = Operation(N, deploy_cost, docked_cost, no_scooter_penalty, pickup_reward, max_scooter_per_area, equity_penalty, rider_test_list, logistics_car_cost_per_mile, logistics_car_capacity, penalty_dist_bus_stop, reward_num_bus_stop)

    # Adding multiple service areas to the study area.
    for loc, geometry, departure_rate, arrival_rate, dist_bus_stop, num_bus_stop in services:
        study_area.add_service(geometry, departure_rate, arrival_rate, loc, max_scooter_per_area, dist_bus_stop, num_bus_stop)
        
    # study_area.add_study_area_rider()
    
    '''
    for service in study_area.map:
        riders = service.add_rider()
        print((1, riders, service.riders))
    '''

    return study_area

def batch_sampling3(x, y, z, batch_size):
    """Randomly pick elements from the three arrays. Elements of the same indexes
    are picked from the three arrays and the pairing will be retained.

    Args:
       x (array): Array of elements.
       y (array): Array of elements with the same length as x.
       z (array): Array of elements with the same length as x.
       batch_size (int): Number of elements to be sampled.

    Returns:
       Three arrays with elements sampled from array x, y and z respectively.

    """
    samples = np.random.randint(len(x), size=batch_size)
    return x[samples], y[samples], z[samples]

###################################################################################
#                       Reinforcement Learning Methods                            #
###################################################################################


def reinforce(study_area, nS, estimator_policy, estimator_value, n_epochs, n_iters, \
              batch_size, display_step, n_test=5):
    """REINFORCE with baseline is implemented to find an optimal policy.
    At certain epochs, the performance of the current policy will be tested.

    Args:
       study area (BusLine): Study area that the algorithm will be implemented on.
       estimator_policy (PolicyEstimator): Policy function approximator to be
                                           optimized.
       estimator_value (StateValueEstimator): State-value function approximator
                                              which will be used as a baseline.
       n_epochs (int): Number of epochs to run.
       n_iters (int): Number of iterations in each epochs during training.
       batch_size (int): Number of samples to used during each training phase.
       display_step (int): Number of epochs to run in between each testing phase.
       n_test (int, optional): Number of iterations during testing. Default is 1000.

    Returns:
       Array containing the average reward in each testing phase.

    """
    avg_reward = []                # Stores the average rewards of each testing phase.
    avg_unmet_demand = []
    avg_met_demand = []
    avg_scooter_num = []
    avg_scooter_loc = []
    avg_gini_index_met = []
    avg_gini_index_unmet = []
    avg_wANNR_scooter = []
    avg_moran_scooter = []
    avg_gini_index_scooter = []
    avg_theil_scooter = []
    avg_EMD_scooter = []
    avg_centroid_scooter = []
    avg_dispersion_scooter = []
    avg_kl_scooter = []
    avg_cost = []
#     test = np.empty(n_test) # Stores the rewards at each time step in testing.
#     unmet_demand_list = np.empty(n_test)
#     met_demand_list = np.empty(n_test)
#     gini_index_met_list = np.empty(n_test)
#     gini_index_unmet_list = np.empty(n_test)
#     scooter_num_list = np.empty(n_test)
#     scooter_loc_list = np.empty(n_test)
    scooter_dist_list = np.empty(nS)
    scooter_geometry_list = [None] * nS # np.empty(nS)
#     scooter_scooter_list = [None] * n_test # np.empty(nS)
#     wANNR_scooter_list = np.empty(n_test)
#     moran_scooter_list = np.empty(n_test)
#     gini_index_scooter_list = np.empty(n_test)
#     theil_scooter_list = np.empty(n_test)
#     EMD_scooter_list = np.empty(n_test)
#     centroid_scooter_list = np.empty(n_test)
#     dispersion_scooter_list = np.empty(n_test)
#     kl_scooter_list = np.empty(n_test)

    # Initialize variables to store information on transition during training.
    states = np.empty((n_iters, nS))
    actions = np.empty(n_iters)
    rewards = np.empty(n_iters)
    
    
    def wANNR(gdf):

        # Coordinates and weights
        # gdf.geometry = gdf.geometry.to_crs(epsg=26917)
        coords = np.array([(geom.x, geom.y) for geom in gdf.geometry])
        weights = gdf['scooters'].values
        tree = cKDTree(coords)

        # Find nearest neighbor distance for each point (excluding self)
        dists, idxs = tree.query(coords, k=2)
        nn_dists = dists[:, 1]/1000  # first is self, second is nearest

        # Weighted average NN distance
        weighted_mean_nn = np.average(nn_dists, weights=weights)

        # Expected NN distance for CSR in 2D (Clark & Evans, 1954)
        area = gdf.unary_union.convex_hull.area/1000000
        density = weights.sum() / area
        expected_nn = 0.5 / np.sqrt(density)

        # Ratio
        ANNR = weighted_mean_nn / expected_nn
        return ANNR
    
    def moran(gdf):
        
        # --- Extract geometry and attribute values ---
        # gdf.geometry = gdf.geometry.to_crs(epsg=26917) # set_crs(epsg=4326).
        coords = np.array([(geom.x, geom.y) for geom in gdf.geometry])
        x = gdf['scooters'].values

        # --- Build inverse distance weight matrix ---
        dist_matrix = cdist(coords, coords) / 1000
        np.fill_diagonal(dist_matrix, np.nan)  # avoid divide-by-zero
        W = 1 / (dist_matrix + 1e-6)
        W = np.nan_to_num(W, nan=0.0)

        # --- Row-standardize weight matrix (optional but common) ---
        W = W / W.sum(axis=1, keepdims=True)

        # --- Calculate Moran’s I manually ---
        x_bar = np.mean(x)
        z = x - x_bar
        n = len(x)

        num = np.sum(W * np.outer(z, z))
        den = np.sum(z**2)
        moran_I = (n / np.sum(W)) * (num / den)
        
        return moran_I
    
    def gini(array):
        array = np.array(array, dtype=np.float64)
        if np.amin(array) < 0:
            raise ValueError("Values cannot be negative")

        array += 1e-10  # Prevent division by zero
        array = np.sort(array)
        n = array.size
        index = np.arange(1, n + 1)
        return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))
    
    def theil(array):
        # Estimate point service areas (e.g., Voronoi, buffers, or fixed area)
        # Here we use fixed circular service area (e.g., 200m buffer)
        x = array
        x = x[x > 0]  # Avoid log(0)
        theil = np.sum((x / x.sum()) * np.log((x / x.sum()) * len(x)))
        return theil
    
    def kl_divergence(gdf, q):
        p, q = np.array(gdf['scooters']), np.array(q)
        if np.sum(p) == 0:
            p[-1] = 1
        p, q = p / p.sum(), q / q.sum()
        return np.sum(rel_entr(p, q))
    
    def EMD(gdf):
        # gdf.geometry = gdf.geometry.to_crs(epsg=26917)
        gdf_all_one = gdf.copy() # all one = A
        gdf_all_one['scooters'] = np.ones(len(gdf_all_one)) * 5
        
        gdf_all_one = gdf_all_one.loc[gdf_all_one['scooters'] > 0]
        if gdf['scooters'].sum() > 0:
            gdf = gdf.loc[gdf['scooters'] > 0]
        
        coords_A = np.array([(geom.x, geom.y) for geom in gdf_all_one.geometry])
        coords_B = np.array([(geom.x, geom.y) for geom in gdf.geometry])
        weights_A = np.array(gdf_all_one['scooters'])
        weights_B = np.array(gdf['scooters'])
        
        D = cdist(coords_A, coords_B) / 1000
        weights_A_norm = weights_A / weights_A.sum()
        if weights_B.sum() > 0:
            weights_B_norm = weights_B / weights_B.sum()
        else:
            weights_B_norm = np.ones(len(weights_B)) / len(weights_B)
        
        emd = ot.emd2(weights_A_norm, weights_B_norm, D)
        return emd
    
    def weighted_centroid(coords, weights):
        weights = np.array(weights)
        coords = np.array(coords)
        cx = np.sum(coords[:, 0] * weights) / np.sum(weights)
        cy = np.sum(coords[:, 1] * weights) / np.sum(weights)
        return np.array([cx, cy])

    def weighted_dispersion(coords, weights, centroid):
        dists_sq = np.sum((coords - centroid)**2, axis=1)
        return np.sqrt(np.sum(dists_sq * weights) / np.sum(weights))
    
    def centroid_dispersion(gdf):
        # Assume you have:
        # coords: (n, 2) array of [x, y] coordinates
        # scooters_A: array of counts for Plan A
        # scooters_B: array of counts for Plan B
        # gdf.geometry = gdf.geometry.to_crs(epsg=26917)
        gdf_all_one = gdf.copy() # all one = A
        gdf_all_one['scooters'] = np.ones(len(gdf_all_one)) * 5
        
        gdf_all_one = gdf_all_one.loc[gdf_all_one['scooters'] > 0]
        if np.sum(gdf['scooters'] > 0) > 0:
            gdf = gdf.loc[gdf['scooters'] > 0]
        
        coords_A = np.array([(geom.x, geom.y) for geom in gdf_all_one.geometry])
        coords_B = np.array([(geom.x, geom.y) for geom in gdf.geometry])
        weights_A = np.array(gdf_all_one['scooters'])
        weights_B = np.array(gdf['scooters'])

        centroid_A = weighted_centroid(coords_A, weights_A)
        centroid_B = weighted_centroid(coords_B, weights_B)

        dispersion_A = weighted_dispersion(coords_A, weights_A, centroid_A)
        dispersion_B = weighted_dispersion(coords_B, weights_B, centroid_B)

        # Compare
        centroid_shift = np.linalg.norm(centroid_B - centroid_A)
        dispersion_change = dispersion_B - dispersion_A
        
        return centroid_shift/1000, dispersion_change/1000

    for epoch in range(n_epochs):
        total = 0
        # rider_init_state = study_area.add_study_area_rider()
        
        for i in range(n_iters):
            # Choose action based on the policy function and take the action.
            cur_state = study_area.get_feature()
            # print((cur_state[0].mean(), cur_state[0].max()))
            action_probs = estimator_policy.predict(cur_state)[0].numpy()
            action_probs = np.round(action_probs, 6)
            # print(action_probs)
            action_probs /= np.sum(action_probs)
            action = np.random.choice(len(action_probs), p=action_probs)
            R, scooter_train_list, unmet_demand, met_demand, departure_rate_list, cost = study_area.take_action(action)

            # Keep track of the transition.
            states[i] = cur_state[0]
            rewards[i] = R
            actions[i] = action

            # Add reward to total after half of the total iterations (steady state)
            if i >= (n_iters // 2):
                total += R

        # Average reward of current policy.
        total /= ((n_iters + 1) //2)
        # total /= n_iters

        # Returns is the total differences between rewards and average reward.
        returns = rewards - total
        # print(rewards)
        # print(returns)
        returns = np.expand_dims(np.cumsum(returns[::-1])[::-1] , axis=1)

        # Sample the transitions.
        bstates, breturns, bactions = batch_sampling3(states, returns, actions, batch_size)
        
        # Run optimization on value estimator
        estimator_value.update(bstates, breturns)
        # Calculate the baseline of these states and get the difference with the returns
        baseline = estimator_value.predict(bstates)
        delta = breturns - baseline
        # Run optimization on policy estimator.
        estimator_policy.update(bstates, delta, bactions)
        
        # Test the current policy and get the average reward per time step.
        if (epoch+1) % display_step == 0:
            # rider_test_state = study_area.add_study_area_rider()
            for j in range(n_test):
                # Get the current state and choose action based on policy function.
                state = study_area.get_feature()
                action_probs = estimator_policy.predict(state)[0].numpy()
                action_probs = np.round(action_probs, 6)
                action_probs /= np.sum(action_probs)
                action = np.random.choice(len(action_probs), p=action_probs)
                #if action > 0:
                #    print("True")
                # print(action)
                test, scooter_list, unmet_demand, met_demand, departure_rate_list, cost = study_area.take_action_test(action)
                # print(cost)
                unmet_demand_list = np.sum(unmet_demand)
                met_demand_list = np.sum(met_demand)
#                 gini_index_unmet_list[j] = gini(unmet_demand)
#                 gini_index_met_list[j] = gini(met_demand)
            for mn in range(nS):
                scooter_dist_list[mn] = scooter_list[mn].scooters_total
                scooter_geometry_list[mn] = scooter_list[mn].geometry
            scooter_gdf = gpd.GeoDataFrame({'scooters': scooter_dist_list,
                                           'geometry': scooter_geometry_list})
            scooter_gdf['geometry'] = scooter_gdf['geometry'].set_crs(epsg=4326).to_crs(epsg=26917)
            # scooter_gdf = scooter_gdf[scooter_gdf['scooters'] > 0]
            scooter_num_list = np.sum(scooter_dist_list)
            scooter_loc_list = np.sum(scooter_dist_list != 0)
            wANNR_scooter_list = wANNR(scooter_gdf)
            moran_scooter_list = moran(scooter_gdf)
            gini_index_scooter_list = gini(scooter_gdf['scooters'])
            theil_scooter_list = theil(scooter_gdf['scooters'])
            EMD_scooter_list = EMD(scooter_gdf)
            centroid_scooter_list, dispersion_scooter_list = centroid_dispersion(scooter_gdf)
            kl_scooter_list = kl_divergence(scooter_gdf, departure_rate_list)
            
            # index = np.argsort(test)[-3]
            avg_reward.append(test)
            avg_unmet_demand.append(unmet_demand_list)
            avg_met_demand.append(met_demand_list)
            avg_scooter_num.append(scooter_num_list)
            avg_scooter_loc.append(scooter_loc_list)
#             avg_gini_index_met.append(gini_index_met_list[index])
#             avg_gini_index_unmet.append(gini_index_unmet_list[index])
            avg_wANNR_scooter.append(wANNR_scooter_list)
            avg_moran_scooter.append(moran_scooter_list)
            avg_gini_index_scooter.append(gini_index_scooter_list)
            avg_theil_scooter.append(theil_scooter_list)
            avg_EMD_scooter.append(EMD_scooter_list)
            avg_centroid_scooter.append(centroid_scooter_list)
            avg_dispersion_scooter.append(dispersion_scooter_list)
            avg_kl_scooter.append(kl_scooter_list)
            avg_cost.append(cost)
            # scooter_list = scooter_scooter_list[index]
#             avg_reward.append(np.mean(test))
#             avg_unmet_demand.append(np.mean(unmet_demand_list))
#             avg_met_demand.append(np.mean(met_demand_list))
#             avg_scooter_num.append(np.mean(scooter_num_list))
#             avg_scooter_loc.append(np.mean(scooter_loc_list))
#             avg_gini_index_met.append(np.mean(gini_index_met_list))
#             avg_gini_index_unmet.append(np.mean(gini_index_unmet_list))
#             avg_wANNR_scooter.append(np.mean(wANNR_scooter_list))
#             avg_moran_scooter.append(np.mean(moran_scooter_list))
#             avg_gini_index_scooter.append(np.mean(gini_index_scooter_list))
#             avg_theil_scooter.append(np.mean(theil_scooter_list))
#             avg_EMD_scooter.append(np.mean(EMD_scooter_list))
#             avg_centroid_scooter.append(np.mean(centroid_scooter_list))
#             avg_dispersion_scooter.append(np.mean(dispersion_scooter_list))
            print("Epoch " + str(epoch+1) + ", Average reward, unmet demand, met demand, scooter num, scooter loc, wANNR, moran, gini, theil, EMD, centroid change, dispersion change, kl divergence, rebalancing cost = " + "{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(avg_reward[-1], avg_unmet_demand[-1], avg_met_demand[-1], avg_scooter_num[-1], avg_scooter_loc[-1], avg_wANNR_scooter[-1], avg_moran_scooter[-1], avg_gini_index_scooter[-1], avg_theil_scooter[-1], avg_EMD_scooter[-1], avg_centroid_scooter[-1], avg_dispersion_scooter[-1], avg_kl_scooter[-1], avg_cost[-1]))

    return avg_reward, avg_unmet_demand, avg_met_demand, avg_scooter_num, avg_scooter_loc, avg_wANNR_scooter, avg_moran_scooter, avg_gini_index_scooter, avg_theil_scooter, avg_EMD_scooter, avg_centroid_scooter, avg_dispersion_scooter, avg_kl_scooter, avg_cost, scooter_list