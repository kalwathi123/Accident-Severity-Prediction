import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from geopy.geocoders import Photon


def augment_data(df):
    bool_features = [
        'Crossing', 'Give_Way', 'Junction', 'No_Exit',
        'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming',
        'Traffic_Signal'
    ]

    for column in bool_features:
        df[column] = np.random.choice([True, False], len(df))

    return df


def encode_feature(df, model_features, feature_to_encode):
    features = [feature for feature in model_features if feature.startswith(feature_to_encode)]
    values = [feature.replace(f"{feature_to_encode}_", "") for feature in features]

    for value in values:
        df[f"{feature_to_encode}_{value}"] = df[feature_to_encode].apply(lambda ele: 1 if ele == value else 0)

    df = df.drop(feature_to_encode, axis=1)
    
    return df


def transform_data(df, model, binary_encoder_pkl, scaler_pkl):

    # Convert the start_time column to datetime format
    df['Start_Time'] = pd.to_datetime(df['Start_Time'])

    # Extract year, month, day, hour, and weekday from the start_time column
    df['Year'] = df['Start_Time'].dt.year
    df['Month'] = df['Start_Time'].dt.month
    df['Day'] = df['Start_Time'].dt.day
    df['Hour'] = df['Start_Time'].dt.hour
    df['Weekday'] = df['Start_Time'].dt.weekday

    # Drop the original start_time column
    df = df.drop('Start_Time', axis=1)

    model_features = model.feature_names_in_.tolist()
    df = encode_feature(df, model_features, "Wind_Direction")
    df = encode_feature(df, model_features, "Weather_Condition")
    df = encode_feature(df, model_features, "Civil_Twilight")

    # Binary Encoding
    with open(binary_encoder_pkl, 'rb') as file:
        binary_encoder = pickle.load(file)

    city_encoded = binary_encoder.transform(df["City"])
    df = pd.concat([df, city_encoded], axis=1).drop("City", axis=1)

    # Label Encoding
    df = df.replace([True, False], [1, 0])

    # Scaler
    features = [
        'Temperature(F)'
        , 'Distance(mi)'
        , 'Humidity(%)'
        , 'Pressure(in)'
        , 'Visibility(mi)'
        , 'Wind_Speed(mph)'
        , 'Start_Lng'
        , 'Start_Lat'
        , 'Year'
        , 'Month'
        , 'Day'
        , 'Hour'
        , 'Weekday'
    ]

    with open(scaler_pkl, 'rb') as file:
        scaler = pickle.load(file)

    df[features] = scaler.transform(df[features])
    return df


# __main__
plot_data = list()

binary_encoder_pkl, scaler_pkl, model_pkl = (
    "C:/Users/ajayv/OneDrive - University at Buffalo/Spring 24/CSE 587 Data Intensive Computing/Project/pickle/model/binary_encoder_city.pkl",
    "C:/Users/ajayv/OneDrive - University at Buffalo/Spring 24/CSE 587 Data Intensive Computing/Project/pickle/model/min_max_scaler.pkl",
    "C:/Users/ajayv/OneDrive - University at Buffalo/Spring 24/CSE 587 Data Intensive Computing/Project/pickle/model/random-forest.pkl"
    )

# Loading the model
if 'model' not in st.session_state:
    with open(model_pkl, 'rb') as file:
        st.session_state['model'] = pickle.load(file)

if 'predict_severity' not in st.session_state:
    st.session_state['predict_severity'] = pd.DataFrame(columns=["Latitude", "Longitude", "City", "Severity"])

st.set_page_config(page_title="Accident Severity", page_icon="")

# Header with logo (replace "logo.png" with your actual logo image path)
st.header("Accident Severity Prediction")
st.image("https://www.800perkins.com/wp-content/uploads/2021/08/What-is-the-First-Thing-I-Should-Do-After-a-Car-Accident-in-Connecticut.png.webp", width=100)

# Tabs for Find Cars and Price History
predict_severity, show_insights = st.tabs(["Predict Severity", "Show Insights"])

with predict_severity:
    # Predict severity tab content
    st.subheader("To Predict the Severity Fill the Below Fields")

    input_data = {}

    # Input fields for user to enter query parameters
    address = st.text_input('Enter Address (e.g., "1600 Amphitheatre Parkway, Mountain View, CA"):', value="1600 Amphitheatre Parkway, Mountain View, CA")

    if address:
        geolocator = Photon(user_agent="geoapiExercises")
        location = geolocator.geocode(address)
        if location:
            latitude = location.latitude
            longitude = location.longitude
            city = location.raw["properties"]["city"]
            input_data["Start_Lat"] = latitude
            input_data["Start_Lng"] = longitude
            input_data["City"] = city
            st.success(f'City: {city}')
        else:
            st.error('Location not found. Please enter a valid address.')
    else:
        st.warning('Please enter an address.')

    input_data["Distance(mi)"] = st.number_input("Distance (mi)", min_value=0., max_value=2., step=0.1)
    input_data["Temperature(F)"] = st.number_input("Temperature (F)", min_value=20, max_value=115, step=1)
    input_data["Humidity(%)"] = st.number_input("Humidity (%)", min_value=1, max_value=100, step=1)
    input_data["Pressure(in)"] = st.number_input("Pressure (in)", min_value=28., max_value=31., step=0.1)
    input_data["Visibility(mi)"] = st.number_input("Visibility (mi)", min_value=1., max_value=10., step=0.1)
    input_data["Wind_Speed(mph)"] = st.number_input("Wind Speed (mph)", min_value=0., max_value=20., step=0.1)

    input_data["Amenity"] = st.radio("Amenity", [True, False])
    input_data["Bump"] = st.radio("Bump", [True, False])

    event_date = st.date_input("Event Date", value="default_value_today", format="YYYY-MM-DD")
    event_time = st.time_input("Event Time", value="now")

    input_data["Start_Time"] = start_time = event_date.strftime('%Y-%m-%d') + " " + event_time.strftime('%H:%M:%S')

    input_data["Weather_Condition"] = st.selectbox("Weather Condition", ["Clear", "Cloudy", "Windy", "Rain", "Thunderstorm", "Fog", "Snow", "Hail", "Sand", "Smoke"])
    input_data["Civil_Twilight"] = st.selectbox("Civil Twilight", ["Day", "Night"])
    input_data["Wind_Direction"] = st.selectbox("Wind Direction", ["N", "S", "E", "W", "Calm", "Variable"])

    input_df = pd.DataFrame([input_data])

    if st.button('Predict'):
        input_df = augment_data(input_df)
        input_df = transform_data(input_df, st.session_state['model'], binary_encoder_pkl, scaler_pkl)

        prediction = st.session_state['model'].predict(input_df[st.session_state['model'].feature_names_in_.tolist()])[0]
        plot_data = [(input_data["Start_Lat"], input_data["Start_Lng"], input_data["City"], prediction)]

        new_df = pd.DataFrame(plot_data, columns=["Latitude", "Longitude", "City", "Severity"])
        st.session_state['predict_severity'] = pd.concat([st.session_state['predict_severity'], new_df], ignore_index=True)

        st.success(f"Severity {prediction}")

with show_insights:
    # Show insignts tab content
    st.subheader("Insights on Accidents Occured")

    # Plot 1
    fig = px.scatter_mapbox(st.session_state['predict_severity'], lat="Latitude", lon="Longitude", hover_name="City", zoom=3, height=600, width=900)
    fig.update_layout(
    mapbox_style="white-bg",
    mapbox_layers=[
        {
            "below": 'traces',
            "sourcetype": "raster",
            "sourceattribution": "United States Geological Survey",
            "source": [
                "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
            ]
        }
        ]
    )
    fig.update_traces(
        marker=go.scattermapbox.Marker(
            size=10,
            color='rgb(0, 161, 155)',
            opacity=1.0
        )
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    st.plotly_chart(fig, use_container_width=True)

    # Plot 2
    severities = list(st.session_state['predict_severity']['Severity'].value_counts().to_dict().items())
    severities.sort(key=lambda ele: ele[0])

    fig, ax = plt.subplots()
    ax.bar(list(map(lambda ele: ele[0], severities)), height=list(map(lambda ele: ele[1], severities)))
    plt.xlabel("Severity")
    plt.ylabel("Counts")
    plt.xlim(0, 5)
    plt.ylim(0, 10)
    plt.xticks([1, 2, 3, 4])
    st.pyplot(fig)
