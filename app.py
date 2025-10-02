import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px  # Import the new library
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
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
    st.write("Welcome! This dashboard provides a comprehensive overview of the Smart Bin project's data science components.")
    st.subheader("Project Overview")
    st.write("""
    - **Live Data:** A physical prototype sends real-time fill-level data to a cloud dashboard.
    - **Historical Analysis:** We analyze a large dataset to understand waste generation patterns.
    - **Predictive Modeling:** A machine learning model forecasts when bins will become full.
    - **Route Optimization:** An algorithm calculates the most efficient collection route for full bins.
    """)
    st.subheader("Dataset at a Glance")
    st.dataframe(df.head())
    st.write(f"The dataset contains **{len(df)}** hourly readings from **{df['bin_id'].nunique()}** simulated smart bins.")

# --- EDA Page (Upgraded with Plotly) ---
elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")
    st.write("These charts are now interactive. You can zoom, pan, and hover over the data.")
    
    st.subheader("Average Bin Fill Percentage by Hour of Day")
    hourly_fill_pattern = df.groupby(['hour_of_day', 'area_type'])['bin_fill_percent'].mean().reset_index()
    fig1 = px.line(hourly_fill_pattern, 
                   x='hour_of_day', 
                   y='bin_fill_percent', 
                   color='area_type',
                   title='Average Bin Fill Percentage by Hour of Day',
                   labels={'hour_of_day': 'Hour of Day', 'bin_fill_percent': 'Average Fill Level (%)'})
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Average Bin Fill Percentage by Day of the Week")
    daily_avg = df.groupby('day_of_week')['bin_fill_percent'].mean().reset_index()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig2 = px.bar(daily_avg, 
                  x='day_of_week', 
                  y='bin_fill_percent',
                  category_orders={"day_of_week": day_order},
                  title='Average Bin Fill Percentage by Day of the Week',
                  labels={'day_of_week': 'Day of the Week', 'bin_fill_percent': 'Average Fill Level (%)'})
    st.plotly_chart(fig2, use_container_width=True)

# --- Predictive Model Page (Upgraded Scatter Plot) ---
# --- Predictive Model Page (Upgraded with Beautified Plot) ---
elif page == "Predictive Model":
    st.title("Predictive Model for Bin Fill Level")
    
    # This part remains the same: train the model
    with st.spinner("Preparing data and training model..."):
        features_to_use = ['hour_of_day', 'day_of_week', 'ward', 'area_type', 'time_since_last_pickup']
        target_variable = 'bin_fill_percent'
        model_df = df[features_to_use + [target_variable]].copy()
        model_df = pd.get_dummies(model_df, columns=['day_of_week', 'ward', 'area_type'], drop_first=True)
        X = model_df.drop(target_variable, axis=1)
        y = model_df[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    st.subheader("Model Performance")
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    col1, col2 = st.columns(2)
    col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}%")
    col2.metric("R-squared (R²) Score", f"{r2:.2f}")

    st.subheader("Interactive Analysis of Model Predictions")
    
    # Create a dataframe for the interactive plot
    plot_data = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    # Calculate the error for coloring
    plot_data['Error'] = abs(plot_data['Actual'] - plot_data['Predicted'])
    
    # Take a random sample to keep the plot fast and clear
    plot_data_sample = plot_data.sample(min(5000, len(plot_data)), random_state=42)
    
    # Create the interactive scatter plot with Plotly Express
    fig = px.scatter(
        plot_data_sample, 
        x='Actual', 
        y='Predicted',
        color='Error',
        color_continuous_scale=px.colors.sequential.Viridis,
        marginal_x='histogram',
        marginal_y='histogram',
        hover_name=plot_data_sample.index,
        hover_data={'Actual': ':.2f', 'Predicted': ':.2f', 'Error': ':.2f'},
        title="Actual vs. Predicted Fill Levels (Colored by Prediction Error)"
    )

    # Add the "Perfect Prediction" line
    fig.add_shape(
        type='line',
        x0=0, y0=0,
        x1=100, y1=100,
        line=dict(color='Red', width=2, dash='dash')
    )

    fig.update_layout(
        xaxis_title='Actual Fill Level (%)',
        yaxis_title='Predicted Fill Level (%)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- Route Optimization Page ---
elif page == "Route Optimization":
    # (The rest of the code for route optimization remains the same as our last stable version)
    st.title("Vehicle Route Optimization")
    if 'route_map' not in st.session_state:
        st.session_state.route_map = None
    if st.button("Calculate Optimized Route for Full Bins"):
        with st.spinner("Finding the most efficient route..."):
            full_bins_sample = df[df['bin_fill_percent'] > 80].sample(10, random_state=42)
            full_bins_sample['demand_liters'] = (full_bins_sample['bin_fill_percent'] / 100) * full_bins_sample['bin_capacity_liters']
            depot_location = pd.DataFrame([{'bin_location_lat': 19.05, 'bin_location_lon': 72.85, 'demand_liters': 0, 'bin_id': 'Depot'}], index=[0])
            route_data = pd.concat([depot_location, full_bins_sample]).reset_index(drop=True)
            data = {}
            data['locations'] = list(zip(route_data['bin_location_lat'], route_data['bin_location_lon']))
            data['demands'] = [int(d) for d in route_data['demand_liters']]
            data['vehicle_capacities'] = [20000]
            data['num_vehicles'] = 1
            data['depot'] = 0
            manager = pywrapcp.RoutingIndexManager(len(data['locations']), data['num_vehicles'], data['depot'])
            routing = pywrapcp.RoutingModel(manager)
            def distance_callback(from_index, to_index):
                from_node, to_node = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
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
            search_parameters.time_limit.FromSeconds(5)
            solution = routing.SolveWithParameters(search_parameters)
            if solution:
                st.success("Optimized route found!")
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
                st.session_state.route_map = m
            else:
                st.error("No solution found!")
                st.session_state.route_map = None
    if st.session_state.route_map:
        st.write("### Optimized Route Map")
        st_folium(st.session_state.route_map, key="route_map_key", width=725, height=500)
    else:
        st.write("Click the button above to calculate and display the route.")
