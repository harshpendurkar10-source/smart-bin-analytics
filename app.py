
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# --- Page Configuration ---
st.set_page_config(page_title="Smart Bin Analytics", layout="wide")

# --- Load and Cache Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('smart_bin_historical_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Exploratory Data Analysis", "Predictive Model", "Route Optimization"])

# --- Home Page ---
if page == "Home":
    st.title("Smart Waste Management Analytics Dashboard")
    st.write("Welcome to the central analytics dashboard for the Smart Bin project. Use the navigation on the left to explore the different data science components.")
    
    st.subheader("Project Overview")
    st.write("""
    This project demonstrates an end-to-end IoT and Data Science solution for waste management.
    - **Live Data:** A physical prototype sends real-time fill-level data to a cloud dashboard.
    - **Historical Analysis:** We analyze a large dataset to understand waste generation patterns.
    - **Predictive Modeling:** A machine learning model forecasts when bins will become full.
    - **Route Optimization:** An algorithm calculates the most efficient collection route for full bins, saving fuel and reducing emissions.
    """)
    
    st.subheader("Dataset at a Glance")
    st.dataframe(df.head())
    st.write(f"The dataset contains **{len(df)}** hourly readings from **{df['bin_id'].nunique()}** simulated smart bins.")

# --- EDA Page ---
elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")
    
    st.subheader("Average Bin Fill Percentage by Hour of Day")
    fig1, ax1 = plt.subplots(figsize=(15, 7))
    hourly_fill_pattern = df.groupby(['hour_of_day', 'area_type'])['bin_fill_percent'].mean().reset_index()
    sns.lineplot(data=hourly_fill_pattern, x='hour_of_day', y='bin_fill_percent', hue='area_type', ax=ax1, lw=3)
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Average Fill Level (%)')
    ax1.set_xticks(range(0, 24))
    ax1.grid(True)
    st.pyplot(fig1)

    st.subheader("Average Bin Fill Percentage by Day of the Week")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sns.barplot(data=df, x='day_of_week', y='bin_fill_percent', order=day_order, palette='viridis', ax=ax2)
    ax2.set_xlabel('Day of the Week')
    ax2.set_ylabel('Average Fill Level (%)')
    st.pyplot(fig2)

# --- Predictive Model Page ---
elif page == "Predictive Model":
    st.title("Predictive Model for Bin Fill Level")
    
    with st.spinner("Preparing data and training model..."):
        # Preprocessing
        features_to_use = ['hour_of_day', 'day_of_week', 'ward', 'area_type', 'time_since_last_pickup']
        target_variable = 'bin_fill_percent'
        model_df = df[features_to_use + [target_variable]].copy()
        model_df = pd.get_dummies(model_df, columns=['day_of_week', 'ward', 'area_type'], drop_first=True)
        
        # Train-test split
        X = model_df.drop(target_variable, axis=1)
        y = model_df[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)

    st.subheader("Model Performance")
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    col1, col2 = st.columns(2)
    col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}%")
    col2.metric("R-squared (R²) Score", f"{r2:.2f}")

    st.subheader("Actual vs. Predicted Values")
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=predictions, alpha=0.5, ax=ax)
    ax.plot([0, 100], [0, 100], color='red', linestyle='--', lw=2)
    ax.set_xlabel('Actual Fill Level (%)')
    ax.set_ylabel('Predicted Fill Level (%)')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    st.pyplot(fig)

# --- Route Optimization Page ---
elif page == "Route Optimization":
    st.title("Vehicle Route Optimization")

    if st.button("Calculate Optimized Route for Full Bins"):
        with st.spinner("Finding the most efficient route..."):
            # Prepare data for solver
            full_bins_sample = df[df['bin_fill_percent'] > 80].sample(10, random_state=42)
            full_bins_sample['demand_liters'] = (full_bins_sample['bin_fill_percent'] / 100) * full_bins_sample['bin_capacity_liters']
            depot_location = pd.DataFrame([{'bin_location_lat': 19.05, 'bin_location_lon': 72.85, 'demand_liters': 0, 'bin_id': 'Depot'}], index=[0])
            route_data = pd.concat([depot_location, full_bins_sample]).reset_index(drop=True)

            data = {}
            data['locations'] = list(zip(route_data['bin_location_lat'], route_data['bin_location_lon']))
            data['demands'] = [int(d) for d in route_data['demand_liters']]
            data['vehicle_capacities'] = [7000]
            data['num_vehicles'] = 1
            data['depot'] = 0

            manager = pywrapcp.RoutingIndexManager(len(data['locations']), data['num_vehicles'], data['depot'])
            routing = pywrapcp.RoutingModel(manager)

            def distance_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return int(abs(data['locations'][from_node][0] - data['locations'][to_node][0]) * 10000 + abs(data['locations'][from_node][1] - data['locations'][to_node][1]) * 10000)

            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

            def demand_callback(from_index):
                from_node = manager.IndexToNode(from_index)
                return data['demands'][from_node]

            demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
            routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, data['vehicle_capacities'], True, 'Capacity')

            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
            search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
            search_parameters.time_limit.FromSeconds(1)

            solution = routing.SolveWithParameters(search_parameters)

            if solution:
                st.success("Optimized route found!")
                
                # Get route and draw map
                optimized_route_indices = []
                index = routing.Start(0)
                while not routing.IsEnd(index):
                    optimized_route_indices.append(manager.IndexToNode(index))
                    index = solution.Value(routing.NextVar(index))
                optimized_route_indices.append(manager.IndexToNode(index))
                
                optimized_route_coords = [data['locations'][i] for i in optimized_route_indices]
                
                m = folium.Map(location=[19.0760, 72.8777], zoom_start=12)
                folium.Marker(location=data['locations'][0], popup='Depot', icon=folium.Icon(color='red', icon='home')).add_to(m)
                for idx, row in route_data.iloc[1:].iterrows():
                    folium.Marker(location=[row['bin_location_lat'], row['bin_location_lon']], popup=f"Bin {row['bin_id']} (Demand: {row['demand_liters']:.0f} L)", icon=folium.Icon(color='blue', icon='trash')).add_to(m)
                folium.PolyLine(locations=optimized_route_coords, color='green', weight=5, opacity=0.8).add_to(m)
                
                # Display map
                st_folium(m, width=725, height=500)
            else:
                st.error("No solution found!")
