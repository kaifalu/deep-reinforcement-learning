import numpy as np
import geopandas as gpd
import ot
from scipy.spatial.distance import cdist
from scipy.special import rel_entr
import pandas as pd

class Service:
    """
    Common base class for all small service areas. 
    Define each service area!!!

    Args:
        max_scooter_per_area (int): Maximum number of scooters that a service area contains.
    """

    def __init__(self, geometry, departure_rate, arrival_rate, max_scooter_per_area, dist_bus_stop, num_bus_stop):
        self.scooters = 0             # Current number of scooters at a small service area: Hyperparameters
        self.scooters_deploy = 0      # The number of scooters deployed at a small service area
        self.riders = 0               # Current riders at a small service area if available, whether to deploy scooters
        self.index = 0
        self.max = max_scooter_per_area
        self.departure_rate = departure_rate
        self.arrival_rate = arrival_rate
        self.dist_bus_stop = dist_bus_stop
        self.num_bus_stop = num_bus_stop
        self.geometry = geometry
        self.scooters_total = 0

    def add_rider(self):
        """
        Add scooters to each small service area randomly based on the scooters' arrival rate unless
        it reaches max.
        
        Remove scooters from each small service area randomly based on the scooters' departure rate unless
        it reaches zeros.

        Returns:
            Total number of scooters at a small service area.
        """
        
        # arrival_rate = self.arrival_rate # scooter daily
        # departure_rate = self.departure_rate # scooter daily

        self.riders = max(0, np.random.poisson(self.departure_rate * 1)) # * 934 days: time window - np.random.poisson(arrival_rate * 1)
        total = self.riders
        return total
    
    def add_scooter(self):
        """
        Add scooters into each service area
        """
        if self.riders > 0 and self.scooters_total < self.max:  #  
            self.scooters_deploy = 1
            self.scooters_total += 1
            self.scooters += 1
        else:
            self.scooters_deploy = 0
        
        return self.scooters_deploy, self.scooters_total
    
    def reduce_scooter(self):
        """
        reduce scooters from each service area
        """
        if self.scooters_total > 0:
            self.scooters_deploy = -1
            self.scooters_total -= 1
            self.scooters -= 1
        
        return self.scooters_deploy, self.scooters_total

    def empty(self):
        """
        Checks if there is a scooter available at a small service area.

        Returns:
            True if there is no scooter, False otherwise.
        """
        return self.scooters == 0
    
    def serve(self):
        """
        Rider coming in to the small service area to use the scooters.

        Returns:
           Total number of riders served.
        """
        
        if self.scooters >= self.riders:
            n = self.riders
            self.scooters = self.scooters - n
        else:
            n = self.scooters
            self.scooters = 0
            
        return n
    
    def unserve(self):
        """
        Rider coming in to the small service area to use the scooters.

        Args:
           scooter (Scooter): The number of scooters within the small service area.

        Returns:
           Total number of riders unserved.
        """
        if self.riders > self.scooters:
            m = self.riders - self.scooters
        else:
            m = 0
        return m
    
    def surplus(self):
        if self.scooters_total > self.riders:
            k = self.scooters_total - self.riders
        else:
            k = 0
        return k
    
    def scooter_count(self):
        return self.scooters_total
    

class Scooter:
    """
    Common base class for all scooters.
    Whether to deploy: self.scooter = 0
    How many scooters to be deployed: self.scooter > 0

    Args:
        deploy_cost (float): Cost of deploying a scooter.
        
        usage_rate (int): Number of riders boarding the bus per unit time.
    """
    def __init__(self, geometry, scooters_total, deploy_cost, docked_cost):
        # self.deployed = False           # Indicates whether to deploy scooters in the service area
        self.deploy_cost = deploy_cost
        self.docked_cost = docked_cost
        # self.scooters_deploy = scooters_deploy 
        self.scooters_total = scooters_total
        self.geometry = geometry

    def deploy(self):
        """
        Deploys the scooter in the service area.

        Returns:
            The cost of deploying the scooter.
        """
        # self.deployed = True
        
        if self.scooters_total > 0:
            mm = -self.scooters_total * self.docked_cost - self.scooters_total * self.deploy_cost
        else:
            mm = 0
        
        return mm
        

class Operation:
    """
    Common base class for the study area and contains N Service Areas and X Scooters.

    Args:
        no_scooter_penalty (float): Cost for no scooter available at each service area.
        pickup_reward (float): Reward for picking up a passenger.
        
    """
    def __init__(self, N, deploy_cost, docked_cost, no_scooter_penalty, pickup_reward, max_scooter_per_area, equity_penalty, rider_test_list, logistics_car_cost_per_mile, logistics_car_capacity, penalty_dist_bus_stop, reward_num_bus_stop):
        self.N = N
        self.map = np.zeros(N, dtype=Service)    # Map indicating locations and distribution of Service
        self.scooter_list = np.zeros(N, dtype=Scooter)                        # List of scooters ready for deployment
        self.scooter_total_list = np.zeros(N)
        self.scooter_deploy_list = np.zeros(N)
        self.deploy_cost = deploy_cost
        self.docked_cost = docked_cost
        self.no_scooter_penalty = no_scooter_penalty
        self.pickup_reward = pickup_reward
        self.max = max_scooter_per_area
        self.equity_penalty = equity_penalty
        self.rider_test_list = rider_test_list
        self.logistics_car_cost_per_mile = logistics_car_cost_per_mile
        self.logistics_car_capacity = logistics_car_capacity
        self.penalty_dist_bus_stop = penalty_dist_bus_stop
        self.reward_num_bus_stop = reward_num_bus_stop
        #self.reward = 0

    def add_service(self, geometry, departure_rate, arrival_rate, index, max_scooter_per_area, dist_bus_stop, num_bus_stop):
        """Adds each service area.

        Args:
            max_scooter_per_area (int): Maximum number of scooters that the service area contains.
        """
                           
        if index >= len(self.map):
            print('index is out of bound.')
        elif isinstance(self.map[index], Service):
            print('Scooters already exist at current service area.')
        else:
            self.map[index] = Service(geometry, departure_rate, arrival_rate, max_scooter_per_area, dist_bus_stop, num_bus_stop)

    def get_feature(self):
        """Gets the current representation of the environment. Information on
        the number of riders and scooter at each service area will be given.

        Returns:
            An array where the 2*N elements indicates the the number of riders/scooters at each of the
            N service area.

        """
        feature = np.zeros(self.N)
        # feature = np.zeros(self.N)

        # To obtain the N elements, the number of riders at all service areas are obtained.
        i = 0
        # total_max = 0   # Total maximum scooters that the study area can have
        total_max = 0
        for service in self.map:
            if isinstance(service, Service):
                feature[i] = service.scooters # service.riders
                # feature[i+self.N] = service.scooters
                total_max += service.max
                # print(feature[i])
                # total_max += loc.max
            i += 1
            
        # feature[0:self.N] /= 1000 # max daily passengers
        # feature[self.N:self.N*2] /= self.max
        feature[:] /= total_max

        # Divides the N elements by the total maximum scooters that the
        # N service areas could contain. Each element indicates the proportion of
        # scooters at that service area out of the total maximum.
        #feature[:] /= total_max

        return feature.reshape((1,self.N))
    
    def add_study_area_rider(self):   # add initial riders for each service area
        for service in self.map:
            riders = service.add_rider()
        return riders

    def move_forward(self):
        """Moving forward the system, which includes riders departing from and arriving at the service area
        and scooters that are deployed moving forward.

        Returns:
            The toal reward received.
        """
        
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
            
            # print(weights_A_norm)
            # print(weights_B_norm)
            emd = ot.emd2(weights_A_norm, weights_B_norm, D)
            return emd
        
        def kl_divergence(gdf, q):
            p, q = np.array(gdf['scooters']), np.array(q)
            if np.sum(p) == 0:
                p[-1] = 1
            p, q = p / p.sum(), q / q.sum()
            return np.sum(rel_entr(p, q))
        
        reward_0 = 0
        iiii = 0
        # total_met_demand = 0
        departure_rate_list = []
        arrival_rate_list = []
        unmet_demand_list = []
        met_demand_list = []
        scooter_surplus_list = []
        total_scooter_count = 0
        total_passenger_list = []
        scooter_dist_list = np.empty(self.N)
        scooter_geometry_list = [None] * self.N
        for service in self.map:
            if isinstance(service, Service):
                # scooters_deploy = service.add_scooter()
                # riders = service.add_rider()
                riders = service.add_rider()
                total_passenger_list.append(riders)
                # print(service.scooter_count())
                self.scooter_list[iiii] = Scooter(service.geometry, service.scooter_count(), self.deploy_cost, self.docked_cost)
                reward_0 += self.scooter_list[iiii].deploy()
                total_scooter_count += service.scooter_count()
                
                scooter_dist_list[iiii] = service.scooters_total
                # print(scooter_dist_list[iiii])
                scooter_geometry_list[iiii] = service.geometry
                
                num_unserve = service.unserve()
                reward_0 -= num_unserve * self.no_scooter_penalty
                unmet_demand_list.append(num_unserve)
                
                scooter_surplus = service.surplus()
                scooter_surplus_list.append(scooter_surplus)
                
                num_serve = service.serve() 
                reward_0 += num_serve * self.pickup_reward
                # total_met_demand += num_serve
                met_demand_list.append(num_serve)
                
                reward_0 -= service.dist_bus_stop * self.penalty_dist_bus_stop
                reward_0 += service.num_bus_stop * self.reward_num_bus_stop
                
                departure_rate_list.append(service.departure_rate)
                arrival_rate_list.append(service.arrival_rate)
                
                # riders = service.add_rider()
                # total_passenger_list.append(riders)
                
                iiii += 1
        
        # generate a dataframe
        scooter_gdf = gpd.GeoDataFrame({'scooters': scooter_dist_list,
                                       'geometry': scooter_geometry_list})
        scooter_gdf['geometry'] = scooter_gdf['geometry'].set_crs(epsg=4326).to_crs(epsg=26917)
        # EMD_scooter_list = EMD(scooter_gdf)
        # reward_0 -= EMD_scooter_list * self.equity_penalty
        # kl_scooter_list = kl_divergence(scooter_gdf, departure_rate_list)
        # reward_0 -= kl_scooter_list * self.equity_penalty
        
        # max service coverage
        scooter_nozero_list = np.sum(scooter_gdf['scooters'] > 0)
        reward_0 += scooter_nozero_list * self.equity_penalty / len(scooter_gdf)
        
        iiii = 0
        prob = arrival_rate_list / np.sum(arrival_rate_list)
        # Draw from Multinomial to fix the total sum
        # arrival_num_list = np.random.multinomial(total_met_demand, prob)
        arrival_num_list = np.random.multinomial(np.sum(np.array(met_demand_list)), prob)
        # scooter_list_arr = []
        for service in self.map:
            if isinstance(service, Service):
                service.scooters += arrival_num_list[iiii]
                # scooter_list_arr.append(service.scooters)
                iiii += 1
        
        iii = 0
        index_pos = []
        move_grid_centroid_pos = []
        move_grid_num_pos = []
        index_neg = []
        move_grid_centroid_neg = []
        move_grid_num_neg = []
        for service in self.map:
            # riders = service.add_rider()
            if isinstance(service, Service):
                #if service.max > 0:
                if service.scooters < service.scooters_total:
                    index_pos.append(iii)
                    move_grid_centroid_pos.append(scooter_gdf.iloc[iii].geometry)
                    move_grid_num_pos.append(service.scooters_total - service.scooters)
                elif service.scooters > service.scooters_total:
                    index_neg.append(iii)
                    move_grid_centroid_neg.append(scooter_gdf.iloc[iii].geometry)
                    move_grid_num_neg.append(service.scooters - service.scooters_total)
            iii += 1
            # print(iii)

        # Build GeoDataFrames 
        # consider car capacity and find all grids that need scooter moving out until capacity
        # and then find grids that need scooter moving in
        # and then start the next logistics
        # print(index_neg)
        neg_gdf = gpd.GeoDataFrame({
            'ind': index_neg,
            'value': move_grid_num_neg,
            'geometry': move_grid_centroid_neg
        })

        pos_gdf = gpd.GeoDataFrame({
            'ind': index_pos,
            'value': move_grid_num_pos,
            'geometry': move_grid_centroid_pos
        })
        
        neg_gdf_sorted = gpd.GeoDataFrame({
            'ind': [],
            'value': [],
            'geometry': []
        })

        pos_gdf_sorted = gpd.GeoDataFrame({
            'ind': [],
            'value': [],
            'geometry': []
        })

        cost = 0
        aa = 0
        bb = 0
        cc = 0
        dd = 0
        # print(np.sum(np.array(met_demand_list)))
        # print(neg_gdf['value'].sum())
        # print(pos_gdf['value'].sum())
        if len(neg_gdf) > 0:
            neg_index_near = neg_gdf[neg_gdf['value'] == neg_gdf['value'].max()].iloc[0]['ind']
            while len(neg_gdf) > 0 and len(pos_gdf) > 0:
                ## if excess scooter in a single grid is larger than capacity
                #print(neg_gdf['value'])
                #print(neg_gdf.loc[neg_gdf['ind'] == neg_index_near])
                # print(neg_gdf['value'].loc[neg_gdf['ind'] == neg_index_near])
                # print(neg_gdf['value'].loc[neg_gdf['ind'] == neg_index_near].value)
                if neg_gdf[neg_gdf['ind'] == neg_index_near]['value'].iloc[0] >= self.logistics_car_capacity:
                    # bb = aa + 1
                    neg_gdf_sorted = pd.concat([neg_gdf_sorted, neg_gdf[neg_gdf['ind'] == neg_index_near]], ignore_index = True)
                    neg_gdf_sorted['value'].iloc[-1] = self.logistics_car_capacity
                    if neg_gdf[neg_gdf['ind'] == neg_index_near]['value'].iloc[0] == self.logistics_car_capacity:
                        neg_gdf = neg_gdf[neg_gdf['ind'] != neg_index_near]
                    else:
                        neg_gdf['value'].loc[neg_gdf['ind'] == neg_index_near] -= self.logistics_car_capacity

                    ## find the positive grid that is nearest to the negative one
                    if len(pos_gdf) == 0:
                        break
                    else:
                        dist_array = np.zeros(len(pos_gdf))
                        for jj in range (len(pos_gdf)):
                            dist_array[jj] = neg_gdf_sorted.iloc[-1].geometry.distance(pos_gdf.iloc[jj].geometry)
                        pos_index_order = np.argsort(dist_array)[0]
                        pos_index_near = pos_gdf.iloc[pos_index_order].ind
                        cost += dist_array[pos_index_order] * 0.00062137

                    ## find the shortest path within positive grid
                    if pos_gdf[pos_gdf['ind'] == pos_index_near]['value'].iloc[0] >= self.logistics_car_capacity:
                        # dd = cc + 1
                        pos_gdf_sorted = pd.concat([pos_gdf_sorted, pos_gdf[pos_gdf['ind'] == pos_index_near]], ignore_index = True)
                        pos_gdf_sorted['value'].iloc[-1] = self.logistics_car_capacity
                        if pos_gdf[pos_gdf['ind'] == pos_index_near]['value'].iloc[0] == self.logistics_car_capacity:
                            pos_gdf = pos_gdf[pos_gdf['ind'] != pos_index_near]
                        else:
                            pos_gdf['value'].loc[pos_gdf['ind'] == pos_index_near] -= self.logistics_car_capacity
                    else:
                        pos_gdf_sorted = pd.concat([pos_gdf_sorted, pos_gdf[pos_gdf['ind'] == pos_index_near]], ignore_index = True)
                        dd = cc
                        while cc == dd:
                            pos_gdf = pos_gdf[pos_gdf['ind'] != pos_index_near]
                            cc = int(pos_gdf_sorted['value'].sum()/self.logistics_car_capacity)
                            if len(pos_gdf) == 0:
                                dd = cc
                                break
                            else:
                                dist_array = np.zeros(len(pos_gdf))
                                for kk in range(len(pos_gdf)):
                                    dist_array[kk] = pos_gdf_sorted.iloc[-1].geometry.distance(pos_gdf.iloc[kk].geometry)
                                pos_index_order = np.argsort(dist_array)[0]
                                pos_index_near = pos_gdf.iloc[pos_index_order].ind
                                cost += dist_array[pos_index_order] * 0.00062137
                                pos_gdf_sorted = pd.concat([pos_gdf_sorted, pos_gdf[pos_gdf['ind'] == pos_index_near]], ignore_index = True)
                                dd = int(pos_gdf_sorted['value'].sum()/self.logistics_car_capacity)
                        ff = pos_gdf_sorted['value'].sum() - dd * self.logistics_car_capacity
                        # print('if positive:', (cc, dd, ff))
                        if cc < dd:
                            if ff > 0:
                                pos_gdf_sorted['value'].iloc[-1] -= ff
                                pos_gdf['value'].loc[pos_gdf['ind'] == pos_index_near] = ff
                            else:
                                pos_gdf = pos_gdf[pos_gdf['ind'] != pos_index_near]

                    ## find the negative grid that is nearest to the positive grid
                    if len(neg_gdf) == 0:
                        break
                    else:
                        dist_array = np.zeros(len(neg_gdf))
                        for mm in range(len(neg_gdf)):
                            dist_array[mm] = pos_gdf_sorted.iloc[-1].geometry.distance(neg_gdf.iloc[mm].geometry)
                        neg_index_order = np.argsort(dist_array)[0]
                        neg_index_near = neg_gdf.iloc[neg_index_order].ind
                        cost += dist_array[neg_index_order] * 0.00062137

                else:
                    # find the shortest path within the negative grids
                    neg_gdf_sorted = pd.concat([neg_gdf_sorted, neg_gdf[neg_gdf['ind'] == neg_index_near]], ignore_index = True)
                    bb = aa
                    while aa == bb:
                        neg_gdf = neg_gdf[neg_gdf['ind'] != neg_index_near]
                        aa = int(neg_gdf_sorted['value'].sum()/self.logistics_car_capacity)
                        if len(neg_gdf) == 0:
                            bb = aa
                            break
                        else:
                            dist_array = np.zeros(len(neg_gdf))
                            for nn in range(len(neg_gdf)):
                                dist_array[nn] = neg_gdf_sorted.iloc[-1].geometry.distance(neg_gdf.iloc[nn].geometry)
                            neg_index_order = np.argsort(dist_array)[0]
                            neg_index_near = neg_gdf.iloc[neg_index_order].ind
                            cost += dist_array[neg_index_order] * 0.00062137
                            neg_gdf_sorted = pd.concat([neg_gdf_sorted, neg_gdf[neg_gdf['ind'] == neg_index_near]], ignore_index = True)

                            bb = int(neg_gdf_sorted['value'].sum()/self.logistics_car_capacity)
                    ee = neg_gdf_sorted['value'].sum() - bb * self.logistics_car_capacity
                    # print('negative path:', (aa, bb, ee))
                    if aa < bb:
                        if ee > 0:
                            neg_gdf_sorted['value'].iloc[-1] -= ee
                            neg_gdf['value'].loc[neg_gdf['ind'] == neg_index_near] = ee
                        else:
                            neg_gdf = neg_gdf[neg_gdf['ind'] != neg_index_near]

                    ## find the positive grid that is nearest to the negative one
                    if len(pos_gdf) == 0:
                        break
                    else:
                        dist_array = np.zeros(len(pos_gdf))
                        for jj in range (len(pos_gdf)):
                            dist_array[jj] = neg_gdf_sorted.iloc[-1].geometry.distance(pos_gdf.iloc[jj].geometry)
                        pos_index_order = np.argsort(dist_array)[0]
                        pos_index_near = pos_gdf.iloc[pos_index_order].ind
                        cost += dist_array[pos_index_order] * 0.00062137

                    ## find the shortest path within positive grid
                    if pos_gdf[pos_gdf['ind'] == pos_index_near]['value'].iloc[0] >= self.logistics_car_capacity:
                        # dd = cc + 1
                        pos_gdf_sorted = pd.concat([pos_gdf_sorted, pos_gdf[pos_gdf['ind'] == pos_index_near]], ignore_index = True)
                        pos_gdf_sorted['value'].iloc[-1] = self.logistics_car_capacity
                        if pos_gdf[pos_gdf['ind'] == pos_index_near]['value'].iloc[0] == self.logistics_car_capacity:
                            pos_gdf = pos_gdf[pos_gdf['ind'] != pos_index_near]
                        else:
                            pos_gdf['value'].loc[pos_gdf['ind'] == pos_index_near] -= self.logistics_car_capacity
                    else:
                        pos_gdf_sorted = pd.concat([pos_gdf_sorted, pos_gdf[pos_gdf['ind'] == pos_index_near]], ignore_index = True)
                        dd = cc
                        while cc == dd:
                            pos_gdf = pos_gdf[pos_gdf['ind'] != pos_index_near]
                            cc = int(pos_gdf_sorted['value'].sum()/self.logistics_car_capacity)
                            if len(pos_gdf) == 0:
                                dd = cc
                                break
                            else:
                                dist_array = np.zeros(len(pos_gdf))
                                for kk in range(len(pos_gdf)):
                                    dist_array[kk] = pos_gdf_sorted.iloc[-1].geometry.distance(pos_gdf.iloc[kk].geometry)
                                pos_index_order = np.argsort(dist_array)[0]
                                pos_index_near = pos_gdf.iloc[pos_index_order].ind
                                cost += dist_array[pos_index_order] * 0.00062137
                                pos_gdf_sorted = pd.concat([pos_gdf_sorted, pos_gdf[pos_gdf['ind'] == pos_index_near]], ignore_index = True)
                                dd = int(pos_gdf_sorted['value'].sum()/self.logistics_car_capacity)
                        ff = pos_gdf_sorted['value'].sum() - dd * self.logistics_car_capacity
                        # print('else positive:', (cc, dd, ff))
                        if cc < dd:
                            if ff > 0:
                                pos_gdf_sorted['value'].iloc[-1] -= ff
                                pos_gdf['value'].loc[pos_gdf['ind'] == pos_index_near] = ff
                            else:
                                pos_gdf = pos_gdf[pos_gdf['ind'] != pos_index_near]

                    ## find the negative grid that is nearest to the positive grid
                    if len(neg_gdf) == 0:
                        break
                    else:
                        dist_array = np.zeros(len(neg_gdf))
                        for mm in range(len(neg_gdf)):
                            dist_array[mm] = pos_gdf_sorted.iloc[-1].geometry.distance(neg_gdf.iloc[mm].geometry)
                        neg_index_order = np.argsort(dist_array)[0]
                        neg_index_near = neg_gdf.iloc[neg_index_order].ind
                        cost += dist_array[neg_index_order] * 0.00062137

            reward_0 -= cost * self.logistics_car_cost_per_mile

            total_pos_num = pos_gdf_sorted['value'].sum()
            total_neg_num = neg_gdf_sorted['value'].sum()
            total_neg_pos_diff = total_neg_num - total_pos_num
            if total_neg_pos_diff > 0:
                for i in range(len(neg_gdf_sorted)):
                    ttt = neg_gdf_sorted['value'].iloc[:i+1].sum()
                    if ttt > total_pos_num:
                        break
                neg_gdf_sorted['value'].iloc[i] = total_pos_num - neg_gdf_sorted['value'].iloc[:i].sum()
                if neg_gdf_sorted['value'].iloc[i] == 0:
                    neg_gdf_sorted = neg_gdf_sorted.iloc[0:i]
                else:
                    neg_gdf_sorted = neg_gdf_sorted.iloc[0:i+1]
            elif total_neg_pos_diff < 0:
                for i in range(len(pos_gdf_sorted)):
                    ttt = pos_gdf_sorted['value'].iloc[:i+1].sum()
                    if ttt > total_neg_num:
                        break
                pos_gdf_sorted['value'].iloc[i] = total_neg_num - pos_gdf_sorted['value'].iloc[:i].sum()
                if pos_gdf_sorted['value'].iloc[i] == 0:
                    pos_gdf_sorted = pos_gdf_sorted.iloc[0:i]
                else:
                    pos_gdf_sorted = pos_gdf_sorted.iloc[0:i+1]

            # negative_scooter_list = []    
            for kkk in range(len(neg_gdf_sorted)):
                self.map[int(neg_gdf_sorted['ind'].iloc[kkk])].scooters -= neg_gdf_sorted['value'].iloc[kkk]
                # negative_scooter_list.append(neg_gdf_sorted['value'].iloc[kkk])
            # positive_scooter_list = []
            for lll in range(len(pos_gdf_sorted)):
                self.map[int(pos_gdf_sorted['ind'].iloc[lll])].scooters += pos_gdf_sorted['value'].iloc[lll]
                
        '''
        iiii = 0
        prob = arrival_rate_list / np.sum(arrival_rate_list)
        # Draw from Multinomial to fix the total sum
        # arrival_num_list = np.random.multinomial(total_met_demand, prob)
        arrival_num_list = np.random.multinomial(np.sum(np.array(met_demand_list)), prob)
        # scooter_list_arr = []
        for service in self.map:
            if isinstance(service, Service):
                service.scooters += arrival_num_list[iiii]
                # scooter_list_arr.append(service.scooters)
                iiii += 1
        
        # print('total_scooter_num:', total_scooter_count)
        # print('total_unmet_demand:', np.sum(np.array(unmet_demand_list)))
        # print('total_scooter_surplus:', np.sum(np.array(scooter_surplus_list)))
        
               
        iiii = 0
        met_demand_list_2 = []
        for service in self.map:
            if isinstance(service, Service):
                service.riders = unmet_demand_list[iiii]
                num_unserve = service.unserve()
                reward_0 -= num_unserve * self.no_scooter_penalty
                unmet_demand_list[iiii] = num_unserve

                num_serve = service.serve() 
                reward_0 += num_serve * self.pickup_reward
                # total_met_demand += num_serve
                met_demand_list[iiii] += num_serve
                met_demand_list_2.append(num_serve)

                iiii += 1
        
        iiii = 0
        arrival_num_list = np.random.multinomial(np.sum(np.array(met_demand_list_2)), prob)
        scooter_list_arr = []
        for service in self.map:
            if isinstance(service, Service):
                service.scooters += arrival_num_list[iiii]
                # scooter_list_arr.append(service.scooters)
                iiii += 1
                # if service.scooters_total > 0:
                    # service.scooters += arrival_num_list[iiii]
                    # service.scooters = service.scooters_total # return original state to state next iteration
                # else:
                    # service.scooters = arrival_num_list[iiii]
                # scooter_list_arr.append(service.scooters)
                # iiii += 1
                
        '''
        # print('total_met_demand', total_met_demand)
        # print('unmet_demand:', np.sum(np.array(unmet_demand_list)))
        # return reward_0, unmet_demand_list,total_passenger_list
        return reward_0, np.array(unmet_demand_list), np.array(met_demand_list), departure_rate_list, cost
        
        # return reward_0

    def take_action(self, n=0):
        """Planners take an action. n=0 is not deploying any scooter and n=1 is to
        deploy one scooter, and n =2 is to deply two scooters...

        Args:
            n (int, optional): Action to be taken. Default 0.

        Returns:
            The total reward received after taking the action.

        """
        reward = 0       

        if n > 0:
            if np.random.rand(1) >= 0.1:
                iii = 0
                for service in self.map:
                    if service.riders > max(service.scooters_total, service.scooters): # and service.riders > 1
                        self.scooter_deploy_list[iii], self.scooter_total_list[iii] = service.add_scooter()
                    elif service.riders < min(service.scooters_total, service.scooters): # 
                        self.scooter_deploy_list[iii], self.scooter_total_list[iii] = service.reduce_scooter()
                    else:
                        self.scooter_deploy_list[iii] = 0

            reward_move_forward, unmet_demand_list, met_demand_list, departure_rate_list, cost = self.move_forward()
            reward += reward_move_forward
            
        else:
            # self.reward += self.move_forward()
            reward_move_forward, unmet_demand_list, met_demand_list, departure_rate_list, cost = self.move_forward()
            reward += reward_move_forward
            
        return reward, self.scooter_list, unmet_demand_list, met_demand_list, departure_rate_list, cost  #self.reward
    
    
    def move_forward_test(self):
        """Moving forward the system, which includes riders departing from and arriving at the service area
        and scooters that are deployed moving forward.

        Returns:
            The toal reward received.
        """
        '''
        np.random.seed(42)
        for service in self.map:
            if isinstance(service, Service):
                service.riders = max(0, np.random.poisson(service.departure_rate * 1))
        '''
        iii = 0
        for service in self.map:
            if isinstance(service, Service):
                service.riders = self.rider_test_list[iii]
                iii += 1
        
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
            
            # print(weights_A_norm)
            # print(weights_B_norm)
            emd = ot.emd2(weights_A_norm, weights_B_norm, D)
            return emd
        
        def kl_divergence(gdf, q):
            p, q = np.array(gdf['scooters']), np.array(q)
            if np.sum(p) == 0:
                p[-1] = 1
            p, q = p / p.sum(), q / q.sum()
            return np.sum(rel_entr(p, q))
        
        reward_0 = 0
        iiii = 0
        # total_met_demand = 0
        departure_rate_list = []
        arrival_rate_list = []
        unmet_demand_list = []
        met_demand_list = []
        scooter_surplus_list = []
        total_scooter_count = 0
        total_passenger_list = []
        scooter_dist_list = np.empty(self.N)
        scooter_geometry_list = [None] * self.N
        for service in self.map:
            if isinstance(service, Service):
                # scooters_deploy = service.add_scooter()
                # riders = service.add_rider()
                self.scooter_list[iiii] = Scooter(service.geometry, service.scooter_count(), self.deploy_cost, self.docked_cost)
                reward_0 += self.scooter_list[iiii].deploy()
                total_scooter_count += service.scooter_count()
                
                scooter_dist_list[iiii] = service.scooters_total
                # print(scooter_dist_list[iiii])
                scooter_geometry_list[iiii] = service.geometry
                
                num_unserve = service.unserve()
                reward_0 -= num_unserve * self.no_scooter_penalty
                unmet_demand_list.append(num_unserve)
                
                scooter_surplus = service.surplus()
                scooter_surplus_list.append(scooter_surplus)
                
                num_serve = service.serve() 
                reward_0 += num_serve * self.pickup_reward
                # total_met_demand += num_serve
                met_demand_list.append(num_serve)
                
                reward_0 -= service.dist_bus_stop * self.penalty_dist_bus_stop
                reward_0 += service.num_bus_stop * self.reward_num_bus_stop
                
                departure_rate_list.append(service.departure_rate)
                arrival_rate_list.append(service.arrival_rate)
                
                # riders = service.add_rider()
                # total_passenger_list.append(riders)
                
                iiii += 1
        
        # generate a dataframe
        scooter_gdf = gpd.GeoDataFrame({'scooters': scooter_dist_list,
                                       'geometry': scooter_geometry_list})
        scooter_gdf['geometry'] = scooter_gdf['geometry'].set_crs(epsg=4326).to_crs(epsg=26917)
        # EMD_scooter_list = EMD(scooter_gdf)
        # reward_0 -= EMD_scooter_list * self.equity_penalty
        # kl_scooter_list = kl_divergence(scooter_gdf, departure_rate_list)
        # reward_0 -= kl_scooter_list * self.equity_penalty
        
        # maximize service coverage
        scooter_nozero_list = np.sum(scooter_gdf['scooters'] > 0)
        reward_0 += scooter_nozero_list * self.equity_penalty / len(scooter_gdf)
        
        iiii = 0
        prob = arrival_rate_list / np.sum(arrival_rate_list)
        # Draw from Multinomial to fix the total sum
        # arrival_num_list = np.random.multinomial(total_met_demand, prob)
        arrival_num_list = np.random.multinomial(np.sum(np.array(met_demand_list)), prob)
        # scooter_list_arr = []
        for service in self.map:
            if isinstance(service, Service):
                service.scooters += arrival_num_list[iiii]
                # scooter_list_arr.append(service.scooters)
                iiii += 1
        
        iii = 0
        index_pos = []
        move_grid_centroid_pos = []
        move_grid_num_pos = []
        index_neg = []
        move_grid_centroid_neg = []
        move_grid_num_neg = []
        for service in self.map:
            # riders = service.add_rider()
            if isinstance(service, Service):
                #if service.max > 0:
                if service.scooters < service.scooters_total:
                    index_pos.append(iii)
                    move_grid_centroid_pos.append(scooter_gdf.iloc[iii].geometry)
                    move_grid_num_pos.append(service.scooters_total - service.scooters)
                elif service.scooters > service.scooters_total:
                    index_neg.append(iii)
                    move_grid_centroid_neg.append(scooter_gdf.iloc[iii].geometry)
                    move_grid_num_neg.append(service.scooters - service.scooters_total)
            iii += 1
            # print(iii)

        # Build GeoDataFrames 
        # consider car capacity and find all grids that need scooter moving out until capacity
        # and then find grids that need scooter moving in
        # and then start the next logistics
        # print(index_neg)
        neg_gdf = gpd.GeoDataFrame({
            'ind': index_neg,
            'value': move_grid_num_neg,
            'geometry': move_grid_centroid_neg
        })

        pos_gdf = gpd.GeoDataFrame({
            'ind': index_pos,
            'value': move_grid_num_pos,
            'geometry': move_grid_centroid_pos
        })
        
        neg_gdf_sorted = gpd.GeoDataFrame({
            'ind': [],
            'value': [],
            'geometry': []
        })

        pos_gdf_sorted = gpd.GeoDataFrame({
            'ind': [],
            'value': [],
            'geometry': []
        })

        cost = 0
        aa = 0
        bb = 0
        cc = 0
        dd = 0
        # print(np.sum(np.array(met_demand_list)))
        # print(neg_gdf['value'].sum())
        # print(pos_gdf['value'].sum())
        if len(neg_gdf) > 0:
            neg_index_near = neg_gdf[neg_gdf['value'] == neg_gdf['value'].max()].iloc[0]['ind']
            while len(neg_gdf) > 0 and len(pos_gdf) > 0:
                ## if excess scooter in a single grid is larger than capacity
                #print(neg_gdf['value'])
                #print(neg_gdf.loc[neg_gdf['ind'] == neg_index_near])
                # print(neg_gdf['value'].loc[neg_gdf['ind'] == neg_index_near])
                # print(neg_gdf['value'].loc[neg_gdf['ind'] == neg_index_near].value)
                if neg_gdf[neg_gdf['ind'] == neg_index_near]['value'].iloc[0] >= self.logistics_car_capacity:
                    # bb = aa + 1
                    neg_gdf_sorted = pd.concat([neg_gdf_sorted, neg_gdf[neg_gdf['ind'] == neg_index_near]], ignore_index = True)
                    neg_gdf_sorted['value'].iloc[-1] = self.logistics_car_capacity
                    if neg_gdf[neg_gdf['ind'] == neg_index_near]['value'].iloc[0] == self.logistics_car_capacity:
                        neg_gdf = neg_gdf[neg_gdf['ind'] != neg_index_near]
                    else:
                        neg_gdf['value'].loc[neg_gdf['ind'] == neg_index_near] -= self.logistics_car_capacity

                    ## find the positive grid that is nearest to the negative one
                    if len(pos_gdf) == 0:
                        break
                    else:
                        dist_array = np.zeros(len(pos_gdf))
                        for jj in range (len(pos_gdf)):
                            dist_array[jj] = neg_gdf_sorted.iloc[-1].geometry.distance(pos_gdf.iloc[jj].geometry)
                        pos_index_order = np.argsort(dist_array)[0]
                        pos_index_near = pos_gdf.iloc[pos_index_order].ind
                        cost += dist_array[pos_index_order] * 0.00062137

                    ## find the shortest path within positive grid
                    if pos_gdf[pos_gdf['ind'] == pos_index_near]['value'].iloc[0] >= self.logistics_car_capacity:
                        # dd = cc + 1
                        pos_gdf_sorted = pd.concat([pos_gdf_sorted, pos_gdf[pos_gdf['ind'] == pos_index_near]], ignore_index = True)
                        pos_gdf_sorted['value'].iloc[-1] = self.logistics_car_capacity
                        if pos_gdf[pos_gdf['ind'] == pos_index_near]['value'].iloc[0] == self.logistics_car_capacity:
                            pos_gdf = pos_gdf[pos_gdf['ind'] != pos_index_near]
                        else:
                            pos_gdf['value'].loc[pos_gdf['ind'] == pos_index_near] -= self.logistics_car_capacity
                    else:
                        pos_gdf_sorted = pd.concat([pos_gdf_sorted, pos_gdf[pos_gdf['ind'] == pos_index_near]], ignore_index = True)
                        dd = cc
                        while cc == dd:
                            pos_gdf = pos_gdf[pos_gdf['ind'] != pos_index_near]
                            cc = int(pos_gdf_sorted['value'].sum()/self.logistics_car_capacity)
                            if len(pos_gdf) == 0:
                                dd = cc
                                break
                            else:
                                dist_array = np.zeros(len(pos_gdf))
                                for kk in range(len(pos_gdf)):
                                    dist_array[kk] = pos_gdf_sorted.iloc[-1].geometry.distance(pos_gdf.iloc[kk].geometry)
                                pos_index_order = np.argsort(dist_array)[0]
                                pos_index_near = pos_gdf.iloc[pos_index_order].ind
                                cost += dist_array[pos_index_order] * 0.00062137
                                pos_gdf_sorted = pd.concat([pos_gdf_sorted, pos_gdf[pos_gdf['ind'] == pos_index_near]], ignore_index = True)
                                dd = int(pos_gdf_sorted['value'].sum()/self.logistics_car_capacity)
                        ff = pos_gdf_sorted['value'].sum() - dd * self.logistics_car_capacity
                        # print('if positive:', (cc, dd, ff))
                        if cc < dd:
                            if ff > 0:
                                pos_gdf_sorted['value'].iloc[-1] -= ff
                                pos_gdf['value'].loc[pos_gdf['ind'] == pos_index_near] = ff
                            else:
                                pos_gdf = pos_gdf[pos_gdf['ind'] != pos_index_near]

                    ## find the negative grid that is nearest to the positive grid
                    if len(neg_gdf) == 0:
                        break
                    else:
                        dist_array = np.zeros(len(neg_gdf))
                        for mm in range(len(neg_gdf)):
                            dist_array[mm] = pos_gdf_sorted.iloc[-1].geometry.distance(neg_gdf.iloc[mm].geometry)
                        neg_index_order = np.argsort(dist_array)[0]
                        neg_index_near = neg_gdf.iloc[neg_index_order].ind
                        cost += dist_array[neg_index_order] * 0.00062137

                else:
                    # find the shortest path within the negative grids
                    neg_gdf_sorted = pd.concat([neg_gdf_sorted, neg_gdf[neg_gdf['ind'] == neg_index_near]], ignore_index = True)
                    bb = aa
                    while aa == bb:
                        neg_gdf = neg_gdf[neg_gdf['ind'] != neg_index_near]
                        aa = int(neg_gdf_sorted['value'].sum()/self.logistics_car_capacity)
                        if len(neg_gdf) == 0:
                            bb = aa
                            break
                        else:
                            dist_array = np.zeros(len(neg_gdf))
                            for nn in range(len(neg_gdf)):
                                dist_array[nn] = neg_gdf_sorted.iloc[-1].geometry.distance(neg_gdf.iloc[nn].geometry)
                            neg_index_order = np.argsort(dist_array)[0]
                            neg_index_near = neg_gdf.iloc[neg_index_order].ind
                            cost += dist_array[neg_index_order] * 0.00062137
                            neg_gdf_sorted = pd.concat([neg_gdf_sorted, neg_gdf[neg_gdf['ind'] == neg_index_near]], ignore_index = True)

                            bb = int(neg_gdf_sorted['value'].sum()/self.logistics_car_capacity)
                    ee = neg_gdf_sorted['value'].sum() - bb * self.logistics_car_capacity
                    # print('negative path:', (aa, bb, ee))
                    if aa < bb:
                        if ee > 0:
                            neg_gdf_sorted['value'].iloc[-1] -= ee
                            neg_gdf['value'].loc[neg_gdf['ind'] == neg_index_near] = ee
                        else:
                            neg_gdf = neg_gdf[neg_gdf['ind'] != neg_index_near]

                    ## find the positive grid that is nearest to the negative one
                    if len(pos_gdf) == 0:
                        break
                    else:
                        dist_array = np.zeros(len(pos_gdf))
                        for jj in range (len(pos_gdf)):
                            dist_array[jj] = neg_gdf_sorted.iloc[-1].geometry.distance(pos_gdf.iloc[jj].geometry)
                        pos_index_order = np.argsort(dist_array)[0]
                        pos_index_near = pos_gdf.iloc[pos_index_order].ind
                        cost += dist_array[pos_index_order] * 0.00062137

                    ## find the shortest path within positive grid
                    if pos_gdf[pos_gdf['ind'] == pos_index_near]['value'].iloc[0] >= self.logistics_car_capacity:
                        # dd = cc + 1
                        pos_gdf_sorted = pd.concat([pos_gdf_sorted, pos_gdf[pos_gdf['ind'] == pos_index_near]], ignore_index = True)
                        pos_gdf_sorted['value'].iloc[-1] = self.logistics_car_capacity
                        if pos_gdf[pos_gdf['ind'] == pos_index_near]['value'].iloc[0] == self.logistics_car_capacity:
                            pos_gdf = pos_gdf[pos_gdf['ind'] != pos_index_near]
                        else:
                            pos_gdf['value'].loc[pos_gdf['ind'] == pos_index_near] -= self.logistics_car_capacity
                    else:
                        pos_gdf_sorted = pd.concat([pos_gdf_sorted, pos_gdf[pos_gdf['ind'] == pos_index_near]], ignore_index = True)
                        dd = cc
                        while cc == dd:
                            pos_gdf = pos_gdf[pos_gdf['ind'] != pos_index_near]
                            cc = int(pos_gdf_sorted['value'].sum()/self.logistics_car_capacity)
                            if len(pos_gdf) == 0:
                                dd = cc
                                break
                            else:
                                dist_array = np.zeros(len(pos_gdf))
                                for kk in range(len(pos_gdf)):
                                    dist_array[kk] = pos_gdf_sorted.iloc[-1].geometry.distance(pos_gdf.iloc[kk].geometry)
                                pos_index_order = np.argsort(dist_array)[0]
                                pos_index_near = pos_gdf.iloc[pos_index_order].ind
                                cost += dist_array[pos_index_order] * 0.00062137
                                pos_gdf_sorted = pd.concat([pos_gdf_sorted, pos_gdf[pos_gdf['ind'] == pos_index_near]], ignore_index = True)
                                dd = int(pos_gdf_sorted['value'].sum()/self.logistics_car_capacity)
                        ff = pos_gdf_sorted['value'].sum() - dd * self.logistics_car_capacity
                        # print('else positive:', (cc, dd, ff))
                        if cc < dd:
                            if ff > 0:
                                pos_gdf_sorted['value'].iloc[-1] -= ff
                                pos_gdf['value'].loc[pos_gdf['ind'] == pos_index_near] = ff
                            else:
                                pos_gdf = pos_gdf[pos_gdf['ind'] != pos_index_near]

                    ## find the negative grid that is nearest to the positive grid
                    if len(neg_gdf) == 0:
                        break
                    else:
                        dist_array = np.zeros(len(neg_gdf))
                        for mm in range(len(neg_gdf)):
                            dist_array[mm] = pos_gdf_sorted.iloc[-1].geometry.distance(neg_gdf.iloc[mm].geometry)
                        neg_index_order = np.argsort(dist_array)[0]
                        neg_index_near = neg_gdf.iloc[neg_index_order].ind
                        cost += dist_array[neg_index_order] * 0.00062137

            reward_0 -= cost * self.logistics_car_cost_per_mile

            total_pos_num = pos_gdf_sorted['value'].sum()
            total_neg_num = neg_gdf_sorted['value'].sum()
            total_neg_pos_diff = total_neg_num - total_pos_num
            if total_neg_pos_diff > 0:
                for i in range(len(neg_gdf_sorted)):
                    ttt = neg_gdf_sorted['value'].iloc[:i+1].sum()
                    if ttt > total_pos_num:
                        break
                neg_gdf_sorted['value'].iloc[i] = total_pos_num - neg_gdf_sorted['value'].iloc[:i].sum()
                if neg_gdf_sorted['value'].iloc[i] == 0:
                    neg_gdf_sorted = neg_gdf_sorted.iloc[0:i]
                else:
                    neg_gdf_sorted = neg_gdf_sorted.iloc[0:i+1]
            elif total_neg_pos_diff < 0:
                for i in range(len(pos_gdf_sorted)):
                    ttt = pos_gdf_sorted['value'].iloc[:i+1].sum()
                    if ttt > total_neg_num:
                        break
                pos_gdf_sorted['value'].iloc[i] = total_neg_num - pos_gdf_sorted['value'].iloc[:i].sum()
                if pos_gdf_sorted['value'].iloc[i] == 0:
                    pos_gdf_sorted = pos_gdf_sorted.iloc[0:i]
                else:
                    pos_gdf_sorted = pos_gdf_sorted.iloc[0:i+1]

            # negative_scooter_list = []    
            for kkk in range(len(neg_gdf_sorted)):
                self.map[int(neg_gdf_sorted['ind'].iloc[kkk])].scooters -= neg_gdf_sorted['value'].iloc[kkk]
                # negative_scooter_list.append(neg_gdf_sorted['value'].iloc[kkk])
            # positive_scooter_list = []
            for lll in range(len(pos_gdf_sorted)):
                self.map[int(pos_gdf_sorted['ind'].iloc[lll])].scooters += pos_gdf_sorted['value'].iloc[lll]
        
        '''
        # return
        iiii = 0
        prob = arrival_rate_list / np.sum(arrival_rate_list)
        # Draw from Multinomial to fix the total sum
        # arrival_num_list = np.random.multinomial(total_met_demand, prob)
        arrival_num_list = np.random.multinomial(np.sum(np.array(met_demand_list)), prob)
        # scooter_list_arr = []
        for service in self.map:
            if isinstance(service, Service):
                service.scooters += arrival_num_list[iiii]
                # scooter_list_arr.append(service.scooters)
                iiii += 1
        
        # print('total_scooter_num:', total_scooter_count)
        # print('total_unmet_demand:', np.sum(np.array(unmet_demand_list)))
        # print('total_scooter_surplus:', np.sum(np.array(scooter_surplus_list)))
              
        iiii = 0
        met_demand_list_2 = []
        for service in self.map:
            if isinstance(service, Service):
                service.riders = unmet_demand_list[iiii]
                num_unserve = service.unserve()
                reward_0 -= num_unserve * self.no_scooter_penalty
                unmet_demand_list[iiii] = num_unserve

                num_serve = service.serve() 
                reward_0 += num_serve * self.pickup_reward
                # total_met_demand += num_serve
                met_demand_list[iiii] += num_serve
                met_demand_list_2.append(num_serve)

                # riders = service.add_rider()
                # total_passenger_list.append(riders)
                iiii += 1
        
        iiii = 0
        arrival_num_list = np.random.multinomial(np.sum(np.array(met_demand_list_2)), prob)
        scooter_list_arr = []
        for service in self.map:
            if isinstance(service, Service):
                service.scooters += arrival_num_list[iiii]
                # scooter_list_arr.append(service.scooters)
                iiii += 1
                # if service.scooters_total > 0:
                    # service.scooters += arrival_num_list[iiii]
                    # service.scooters = service.scooters_total # return original state to state next iteration
                # else:
                    # service.scooters = arrival_num_list[iiii]
                # scooter_list_arr.append(service.scooters)
                # iiii += 1
        '''        
                
        # print('total_met_demand', total_met_demand)
        # print('unmet_demand:', np.sum(np.array(unmet_demand_list)))
        # return reward_0, unmet_demand_list,total_passenger_list
        return reward_0, np.array(unmet_demand_list), np.array(met_demand_list), departure_rate_list, cost
        
        # return reward_0

    def take_action_test(self, n=0):
        """Planners take an action. n=0 is not deploying any scooter and n=1 is to
        deploy one scooter, and n =2 is to deply two scooters...

        Args:
            n (int, optional): Action to be taken. Default 0.

        Returns:
            The total reward received after taking the action.

        """
        reward = 0       

        if n > 0:
            if np.random.rand(1) >= 0.1:
                iii = 0
                for service in self.map:
                    if service.riders > max(service.scooters_total, service.scooters): # and service.riders > 1
                        self.scooter_deploy_list[iii], self.scooter_total_list[iii] = service.add_scooter()
                    elif service.riders < min(service.scooters_total, service.scooters): # 
                        self.scooter_deploy_list[iii], self.scooter_total_list[iii] = service.reduce_scooter()
                    else:
                        self.scooter_deploy_list[iii] = 0

            reward_move_forward, unmet_demand_list, met_demand_list, departure_rate_list, cost = self.move_forward_test()
            reward += reward_move_forward
            
        else:
            # self.reward += self.move_forward()
            reward_move_forward, unmet_demand_list, met_demand_list, departure_rate_list, cost = self.move_forward_test()
            reward += reward_move_forward
            
        return reward, self.scooter_list, unmet_demand_list, met_demand_list, departure_rate_list, cost  #self.reward
