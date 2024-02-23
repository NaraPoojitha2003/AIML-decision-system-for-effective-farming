import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
from PIL import Image
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
import joblib

# Set page config first
st.set_page_config(
    page_title="AI-ML Decision System for Effective Farming",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Home page
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

# Define sidebar options
sidebar_options = ["Home Page", "Crop Recommendation", "Fertilizer Recommendation", "Crop Yield Prediction"]
page_selection = st.sidebar.radio("Go to", sidebar_options)

# Update session state based on sidebar selection
if page_selection == "Crop Recommendation":
    st.session_state['page'] = 'crop_recommendation'
elif page_selection == "Fertilizer Recommendation":
    st.session_state['page'] = 'fertilizer_recommendation'
elif page_selection == "Crop Yield Prediction":
    st.session_state['page'] = 'crop_yield_prediction'
else:
    st.session_state['page'] = 'home'

# Home page
if st.session_state['page'] == 'home':
    st.write("# AI-ML Decision System for Effective Farming")
    st.image("https://media.istockphoto.com/id/1280116053/vector/illustration-of-a-farmer-sitting-under-a-tree-looking-at-his-laptop-in-the-agricultural-field.jpg?s=612x612&w=0&k=20&c=jQ7UNR-khaS-gXxAlXnuEWrSon3LuFC3n1OB7nTDFJo=", use_column_width=True)

# Crop recommendation page
if 'page' in st.session_state and st.session_state['page'] == 'crop_recommendation':
    
    page_bg = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
    background-color:#90EE90; /* changed to #90EE90 */

    }}
    [data-testid="stSidebar"] {{
    background-color:#8F9779; /* unchanged */

    }}
    [data-testid="stHeader"] {{
    background-color:#90EE90; /* changed to #90EE90 */
    }}
    [data-testid="stToolbar"] {{
    background-color:#90EE90; /* changed to #90EE90 */

    }}
    </style>
    """
    st.markdown(page_bg,unsafe_allow_html=True)

    def load_bootstrap():
            return st.markdown("""<link rel="stylesheet" 
            href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" 
            integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" 
            crossorigin="anonymous">""", unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; color: black;'>Crop Recommendation System</h1>", unsafe_allow_html=True)

    colx, coly, colz = st.columns([1,4,1], gap = 'medium')

    df = pd.read_csv('croprecommendation.csv')
    rdf_clf = joblib.load('final_rdf_clf_climate.pkl')

    X = df.drop('label', axis = 1)
    y = df['label']

    df_desc = pd.read_csv('Crop_Desc.csv', sep = ';', encoding = 'utf-8', encoding_errors = 'ignore')

    st.markdown("<h5 style='text-align: center;'>Importance of each Feature in the Model:</h5>", unsafe_allow_html=True)

    importance = pd.DataFrame({'Feature': list(X.columns),
                   'Importance(%)': rdf_clf.feature_importances_}).\
                    sort_values('Importance(%)', ascending = True)
    importance['Importance(%)'] = importance['Importance(%)'] * 100

    colx, coly, colz = st.columns([1,4,1], gap = 'medium')
    with coly:
        color_discrete_sequence = '#609cd4'
        fig = px.bar(importance , x = 'Importance(%)', y = 'Feature', orientation= 'h', width = 200, height = 300)
        fig.update_traces(marker_color="#8C564B")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width= True)

    st.text("Now insert the values and the system will predict the best crop to plant.")
    st.text("In the (?) marks you can get some help about each feature.")

    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1,1,2,1,2,1,1,2], gap = 'medium')

    with col3:
        n_input = st.number_input('Insert N (kg/ha) value:', min_value= 0, max_value= 140, help = 'Insert here the Nitrogen density (kg/ha) from 0 to 140.')
        p_input = st.number_input('Insert P (kg/ha) value:', min_value= 5, max_value= 145, help = 'Insert here the Phosphorus density (kg/ha) from 5 to 145.')
        k_input = st.number_input('Insert K (kg/ha) value:', min_value= 5, max_value= 205, help = 'Insert here the Potassium density (kg/ha) from 5 to 205.')
        temp_input = st.number_input('Insert Avg Temperature (ºC) value:', min_value= 9., max_value= 43., step = 1., format="%.2f", help = 'Insert here the Avg Temperature (ºC) from 9 to 43.')

    with col5:
        hum_input = st.number_input('Insert Avg Humidity (%) value:', min_value= 15., max_value= 99., step = 1., format="%.2f", help = 'Insert here the Avg Humidity (%) from 15 to 99.')
        ph_input = st.number_input('Insert pH value:', min_value= 3.6, max_value= 9.9, step = 0.1, format="%.2f", help = 'Insert here the pH from 3.6 to 9.9')
        rain_input = st.number_input('Insert Avg Rainfall (mm) value:', min_value= 21.0, max_value= 2700.0, step = 0.1, format="%.2f", help = 'Insert here the Avg Rainfall (mm) from 21 to 2700')

    with col7:
        climate = [0, 1, 2, 3, 4]
        climate_input = st.selectbox('Select Climate:', options=range(len(climate)), format_func=lambda x: climate[x], help="Select the climate type: 0 : Winter, 1 : Pre-Monsoon (Spring), 2 : Monsoon, 3 : Post-Monsoon (Autumn), 4 : Summer")

    predict_inputs = [[n_input,p_input,k_input,temp_input,hum_input,ph_input,rain_input, climate_input]]

    with col5:
        predict_btn = st.button('Get Your Recommendation!')

    cola,colb,colc = st.columns([2,10,2])
    if predict_btn:
        rdf_predicted_value = rdf_clf.predict(predict_inputs)
        #st.text('Crop suggestion: {}'.format(rdf_predicted_value[0]))
        with colb:
            st.markdown(f"<h3 style='text-align: center;'>Best Crop to Plant: {rdf_predicted_value[0]}.</h3>", 
            unsafe_allow_html=True)
        col1, col2, col3 = st.columns([9,4,9])
        with col2:
            df_desc = df_desc.astype({'label':str,'image':str})
            df_desc['label'] = df_desc['label'].str.strip()
            df_desc['image'] = df_desc['image'].str.strip()

            df_pred_image = df_desc[df_desc['label'].isin(rdf_predicted_value)]
            df_image = df_pred_image['image'].item()

            st.markdown(f"""<h5 style = 'text-align: center; height: 300px; object-fit: contain;'> {df_image} </h5>""", unsafe_allow_html=True)
