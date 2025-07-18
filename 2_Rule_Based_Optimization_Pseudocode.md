# Rule-Based Optimization Pseudocode
## Trailer Allocation and Relocation System

### ⚠️ PROJECT DISCLAIMER
This repository is based on a real-world project, but:
- All data, business rules, and parameters have been modified.
- No actual company data or proprietary logic is included.
- Core optimization and architecture are abstracted for illustrative purposes.
- All quantitative aspects have been altered for confidentiality.

---

### Project Overview
This pseudocode demonstrates a comprehensive rule-based optimization system I developed for allocating trailers to network locations and relocating them from their current positions to optimal locations. The project showcases my ability to tackle complex logistics optimization problems using algorithmic thinking, mathematical modeling, and practical business rule implementation.

**Key Problem Solved**: Optimizing trailer distribution across a nationwide network while minimizing relocation costs and maximizing operational efficiency.

---

## 1. MAIN ALGORITHM

```
MAIN()
    SETUP_ENVIRONMENT()
    
    // Project Configuration
    num_cities = 25                    // Target market cities
    trailer_count = 500                // Fleet size to optimize
    
    // Load input data
    city_centers_data = LOAD("MarketIndex_Comdist_Synthetic.csv")
    network_locations_data = LOAD("NetLoc_Synthetic_cleaned.csv")
    telematics_data = LOAD("Telematics_Synthetic.csv")
    
    // Phase 1: Network Allocations
    allocations_df, city_centers_df = PERFORM_NETWORK_ALLOCATIONS(
        city_centers_data, 
        network_locations_data, 
        num_cities, 
        trailer_count
    )
    
    // Phase 2: Trailer Relocations
    final_allocation, total_distance = PERFORM_TRAILER_RELOCATIONS(
        telematics_data, 
        allocations_df, 
        num_cities
    )
    
    PRINT("Optimization completed successfully!")
END
```

---

## 2. NETWORK ALLOCATIONS PHASE

### 2.1 Main Network Allocation Function
```
PERFORM_NETWORK_ALLOCATIONS(city_centers_raw_df, network_locations_df, num_cities, trailer_count)
    // Filter top N cities based on demand
    city_centers_df = SELECT_TOP_N_CITIES(city_centers_raw_df, num_cities)
    
    // Calculate market index percentages
    total_index = SUM(city_centers_df['Index'])
    FOR each city IN city_centers_df:
        city['Percentage'] = city['Index'] / total_index
    
    // Assign tiers and costs based on market index (Business Logic Implementation)
    FOR each city IN city_centers_df:
        IF city['Index'] > 150:                    // High-demand market threshold
            city['tier'] = 'tier1'
            city['daily_revenue'] = 75             // Premium market rates
            city['daily_parking_cost'] = 8         // Higher facility costs
        ELSE IF city['Index'] > 80:                // Medium-demand market threshold
            city['tier'] = 'tier2'
            city['daily_revenue'] = 45             // Standard market rates
            city['daily_parking_cost'] = 6         // Standard facility costs
        ELSE:
            city['tier'] = 'tier3'
            city['daily_revenue'] = 25             // Basic market rates
            city['daily_parking_cost'] = 4         // Basic facility costs
    
    // Calculate trailer demand for each city
    FOR each city IN city_centers_df:
        city['num_trailers'] = ROUND(city['Percentage'] * trailer_count)
    
    // Balance total trailer count
    total_demand = SUM(city_centers_df['num_trailers'])
    difference = total_demand - trailer_count
    
    IF difference > 0:
        // Remove trailers from cities with lowest demand
        sorted_cities = SORT_BY_ASCENDING(city_centers_df['num_trailers'])
        FOR i = 1 TO difference:
            city_index = sorted_cities[i % length(sorted_cities)]
            city_centers_df[city_index]['num_trailers'] -= 1
    ELSE IF difference < 0:
        // Add trailers to cities with lowest demand
        sorted_cities = SORT_BY_ASCENDING(city_centers_df['num_trailers'])
        FOR i = 1 TO ABS(difference):
            city_index = sorted_cities[i % length(sorted_cities)]
            city_centers_df[city_index]['num_trailers'] += 1
    
    // Calculate financial metrics (Business Performance Modeling)
    utilization_days = 22                           // 73% utilization rate
    unutilized_days = 30 - utilization_days
    
    FOR each city IN city_centers_df:
        city['monthly_revenue'] = city['num_trailers'] * city['daily_revenue'] * utilization_days
        city['monthly_parking_cost'] = city['num_trailers'] * city['daily_parking_cost'] * unutilized_days
    
    // Perform allocations for each city
    all_allocations = []
    FOR each city IN city_centers_df:
        city_allocations = ALLOCATE_TRAILERS_FOR_CITY(
            city['Latitude'], 
            city['Longitude'], 
            city['num_trailers'], 
            network_locations_df
        )
        all_allocations.APPEND(city_allocations)
    
    RETURN all_allocations, city_centers_df
END
```

### 2.2 City-Specific Trailer Allocation
```
ALLOCATE_TRAILERS_FOR_CITY(city_lat, city_lon, num_trailers, network_locations_df)
    // Calculate distances from city to all network locations
    FOR each location IN network_locations_df:
        location['distance_miles'] = HAVERSINE_DISTANCE(
            city_lat, city_lon, 
            location['Latitude'], location['Longitude']
        )
    
    // Initialize allocation map
    allocation_map = {}
    FOR each location IN network_locations_df:
        allocation_map[location['Net_loc_id']] = {
            'Net_loc_id': location['Net_loc_id'],
            'Net_loc_name': location['Net_loc_name'],
            'type': location['type'],
            'distance_miles': location['distance_miles'],
            'allocated_trailers': 0
        }
    
    leftover_trailers = num_trailers
    
    // RULE 1: Prioritize Dealer locations within 20 miles (Strategic Priority)
    dealers_within_20 = FILTER_BY_TYPE_AND_DISTANCE(
        network_locations_df, 
        type='Dealer', 
        max_distance=20
    )
    
    IF dealers_within_20 IS NOT EMPTY:
        FOR each dealer IN dealers_within_20:
            IF leftover_trailers <= 0:
                BREAK
            trailers_to_allocate = MIN(20, leftover_trailers)    // Capacity constraint
            allocation_map[dealer['Net_loc_id']]['allocated_trailers'] += trailers_to_allocate
            leftover_trailers -= trailers_to_allocate
    
    // RULE 2: If still have trailers, allocate to Own locations within 20 miles
    IF leftover_trailers > 0:
        outposts_within_20 = FILTER_BY_TYPE_AND_DISTANCE(
            network_locations_df, 
            type='Own', 
            max_distance=20
        )
        
        IF outposts_within_20 IS NOT EMPTY:
            FOR each outpost IN outposts_within_20:
                IF leftover_trailers <= 0:
                    BREAK
                trailers_to_allocate = MIN(20, leftover_trailers)
                allocation_map[outpost['Net_loc_id']]['allocated_trailers'] += trailers_to_allocate
                leftover_trailers -= trailers_to_allocate
    
    // RULE 3: If no Dealer/Own locations within 20 miles, use Own locations
    IF dealers_within_20 IS EMPTY AND leftover_trailers > 0:
        outposts_within_20 = FILTER_BY_TYPE_AND_DISTANCE(
            network_locations_df, 
            type='Own', 
            max_distance=20
        )
        
        IF outposts_within_20 IS NOT EMPTY:
            FOR each outpost IN outposts_within_20:
                IF leftover_trailers <= 0:
                    BREAK
                trailers_to_allocate = MIN(20, leftover_trailers)
                allocation_map[outpost['Net_loc_id']]['allocated_trailers'] += trailers_to_allocate
                leftover_trailers -= trailers_to_allocate
    
    // RULE 4: Allocate remaining trailers to closest locations (any type)
    IF leftover_trailers > 0:
        sorted_locations = SORT_BY_DISTANCE(network_locations_df)
        FOR each location IN sorted_locations:
            IF leftover_trailers <= 0:
                BREAK
            current_allocation = allocation_map[location['Net_loc_id']]['allocated_trailers']
            max_additional = 20 - current_allocation
            trailers_to_allocate = MIN(max_additional, leftover_trailers)
            allocation_map[location['Net_loc_id']]['allocated_trailers'] += trailers_to_allocate
            leftover_trailers -= trailers_to_allocate
    
    RETURN VALUES(allocation_map)
END
```

---

## 3. TRAILER RELOCATIONS PHASE

### 3.1 Main Relocation Function
```
PERFORM_TRAILER_RELOCATIONS(telematics_df, allocations_df, num_cities)
    // Data preprocessing
    telematics_df = CLEAN_TELEMATICS_DATA(telematics_df)
    
    // Get latest position for each trailer
    latest_telematics = GET_LATEST_POSITIONS(telematics_df)
    
    // Validate trailer count consistency
    num_trailers = COUNT(latest_telematics)
    total_required = SUM(allocations_df['allocated_trailers'])
    
    IF total_required != num_trailers:
        RAISE_ERROR("Trailer count mismatch")
    
    // Create location mappings
    net_loc_lat = CREATE_DICT(allocations_df['Net_loc_id'], allocations_df['Latitude_network'])
    net_loc_lon = CREATE_DICT(allocations_df['Net_loc_id'], allocations_df['Longitude_network'])
    net_loc_name = CREATE_DICT(allocations_df['Net_loc_id'], allocations_df['Net_loc_name'])
    net_loc_demand = CREATE_DICT(allocations_df['Net_loc_id'], allocations_df['allocated_trailers'])
    
    // Calculate distance matrix
    costs = {}
    assets = GET_ASSET_LIST(latest_telematics)
    locations = GET_LOCATION_LIST(allocations_df)
    
    FOR each asset IN assets:
        asset_lat = GET_ASSET_LATITUDE(asset)
        asset_lon = GET_ASSET_LONGITUDE(asset)
        FOR each location IN locations:
            loc_lat = net_loc_lat[location]
            loc_lon = net_loc_lon[location]
            distance = HAVERSINE_DISTANCE(asset_lat, asset_lon, loc_lat, loc_lon)
            costs[(asset, location)] = distance
    
    // Solve optimization problem
    solution = SOLVE_ASSIGNMENT_PROBLEM(assets, locations, costs, net_loc_demand)
    
    // Process results and calculate costs (Cost Optimization)
    total_distance = CALCULATE_TOTAL_DISTANCE(solution, costs)
    transportation_cost = total_distance * 1.5  // Operational cost per mile
    
    RETURN solution, total_distance
END
```

### 3.2 Assignment Problem Solver
```
SOLVE_ASSIGNMENT_PROBLEM(assets, locations, costs, net_loc_demand)
    // Initialize OR-Tools solver
    solver = CREATE_SOLVER('SCIP')
    
    // Create decision variables
    x_vars = {}
    FOR each asset IN assets:
        FOR each location IN locations:
            x_vars[(asset, location)] = CREATE_BOOL_VARIABLE(solver)
    
    // Set objective: minimize total distance
    objective = solver.Objective()
    objective.SetMinimization()
    FOR each (asset, location) IN costs:
        objective.SetCoefficient(x_vars[(asset, location)], costs[(asset, location)])
    
    // CONSTRAINT 1: Each asset can only go to 1 location
    FOR each asset IN assets:
        constraint = solver.AddConstraint()
        FOR each location IN locations:
            constraint.SetCoefficient(x_vars[(asset, location)], 1)
        constraint.SetBounds(1, 1)  // Exactly 1 location per asset
    
    // CONSTRAINT 2: Each location must receive its required number of trailers
    FOR each location IN locations:
        constraint = solver.AddConstraint()
        FOR each asset IN assets:
            constraint.SetCoefficient(x_vars[(asset, location)], 1)
        constraint.SetBounds(net_loc_demand[location], net_loc_demand[location])
    
    // Solve the problem
    status = solver.Solve()
    
    IF status != OPTIMAL:
        RAISE_ERROR("No optimal solution found")
    
    // Extract solution
    solution = []
    FOR each asset IN assets:
        FOR each location IN locations:
            IF x_vars[(asset, location)].solution_value() == 1:
                solution.APPEND({
                    'asset_id': asset,
                    'location_id': location,
                    'distance': costs[(asset, location)]
                })
                BREAK
    
    RETURN solution
END
```

---

## 4. UTILITY FUNCTIONS

### 4.1 Distance Calculation
```
HAVERSINE_DISTANCE(lat1, lon1, lat2, lon2)
    // Convert to radians
    lat1_rad = CONVERT_TO_RADIANS(lat1)
    lon1_rad = CONVERT_TO_RADIANS(lon1)
    lat2_rad = CONVERT_TO_RADIANS(lat2)
    lon2_rad = CONVERT_TO_RADIANS(lon2)
    
    // Calculate differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    // Haversine formula
    a = SIN(dlat/2)^2 + COS(lat1_rad) * COS(lat2_rad) * SIN(dlon/2)^2
    c = 2 * ASIN(SQRT(a))
    
    // Earth radius in miles
    r = 3958.7613
    
    RETURN c * r
END
```

### 4.2 Data Filtering
```
FILTER_BY_TYPE_AND_DISTANCE(network_locations_df, type, max_distance)
    filtered_locations = []
    FOR each location IN network_locations_df:
        IF location['type'] == type AND location['distance_miles'] <= max_distance:
            filtered_locations.APPEND(location)
    
    RETURN SORT_BY_DISTANCE(filtered_locations)
END
```

---

## 5. RULE SUMMARY

### Allocation Rules (Priority Order):
1. **Dealer Priority**: Allocate to Dealer locations within 20 miles first (Strategic Partnership Focus)
2. **Own Location Backup**: If no dealers, use Own locations within 20 miles
3. **Own Location Primary**: If no dealers exist, use Own locations within 20 miles
4. **Distance-Based Fallback**: Allocate remaining trailers to closest locations
5. **Capacity Limit**: Maximum 20 trailers per location (Operational Constraint)

### Optimization Rules:
1. **One-to-One Assignment**: Each trailer assigned to exactly one location
2. **Demand Satisfaction**: Each location receives exactly its required number of trailers
3. **Distance Minimization**: Minimize total relocation distance
4. **Cost Calculation**: $1.50 per mile transportation cost (Operational Efficiency)

### Financial Rules (Business Performance Modeling):
1. **Revenue Calculation**: Daily revenue × utilization days (22 days)
2. **Parking Cost**: Daily parking cost × unutilized days (8 days)
3. **Profit Formula**: Revenue - Parking Cost - Transportation Cost
4. **Tier-Based Pricing**: Different rates based on market index tiers
   - Tier 1: $75/day revenue, $8/day parking (Index > 150)
   - Tier 2: $45/day revenue, $6/day parking (Index > 80)
   - Tier 3: $25/day revenue, $4/day parking (Index ≤ 80)

### Project Impact & Skills Demonstrated:
- **Complex Problem Solving**: Multi-phase optimization with business constraints
- **Algorithmic Thinking**: Rule-based allocation with mathematical optimization
- **Business Acumen**: Integration of operational constraints and financial modeling
- **Technical Implementation**: Data processing, geospatial calculations, and optimization algorithms
- **System Design**: Scalable architecture handling multiple data sources and constraints

### ⚠️ PROJECT NOTE
**This pseudocode represents a real logistics optimization project I developed, demonstrating my ability to solve complex business problems through algorithmic thinking and mathematical modeling. While specific numerical values have been modified to protect proprietary information, the core problem-solving approach, technical methodology, and business logic implementation remain representative of the actual project scope and complexity.** 