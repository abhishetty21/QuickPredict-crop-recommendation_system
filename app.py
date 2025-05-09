import streamlit as st
import pandas as pd
import pickle as pi
st.set_page_config(page_title='Qpredict', page_icon=":tractor:")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Load models
models = {}
algo = ['Naive Bayes', 'Decision Tree', 'Random Forest']
for name in algo:
    fn = name + ".prediction"
    md = pi.load(open(fn, 'rb'))
    models[name] = md

# Language selection
lang_options = {'English': 'en', 'Hindi': 'hi', 'Kannada': 'kn'}
selected_lang = st.sidebar.selectbox('Select Language / भाषा चुनें / ಭಾಷೆ ಆಯ್ಕೆ', list(lang_options.keys()))

# Translations
translations = {
    'en': {
        'title': 'Quick Predict',
        'tagline': 'A Crop Recommendation System',
        'about': 'About',
        'about_info': 'The Crop Recommendation System helps farmers choose the most suitable crop based on soil and weather conditions. This system uses machine learning models to predict the best crop. It was developed by the students (B.E) Abhishek .A. Shetty, Anish PK, Hariprasanna, Sagar .S. Shetty, and Vismay Shetty, under the guidance of Megha Rani R, Assistant Professor, Department of AI & DS, SMVITM.',
        'input_params': 'Input Parameters',
        'nitrogen': 'Nitrogen (N)',
        'phosphorus': 'Phosphorus (P)',
        'potassium': 'Potassium (K)',
        'temperature': 'Temperature (°C)',
        'humidity': 'Humidity (%)',
        'ph': 'pH',
        'rainfall': 'Rainfall (mm)',
        'Recommend': 'Recommend',
        'recommended_crop': 'Recommended Crop',
        'enter_inputs': 'Please enter all input parameters.'
    },
    'hi': {
        'title': 'क्विक प्रेडिक्ट',
        'tagline': 'एक फसल सिफारिश प्रणाली',
        'about': 'के बारे में',
        'about_info': 'फसल सिफारिश प्रणाली किसानों को मिट्टी और मौसम की स्थिति के आधार पर सबसे उपयुक्त फसल चुनने में मदद करती है। यह प्रणाली सबसे अच्छी फसल की भविष्यवाणी करने के लिए मशीन लर्निंग मॉडल का उपयोग करती है। इसे (B.E) अभिषेक .ए. शेट्टी, अनिश पी.के., हरिप्रसन्न, सागर .एस. शेट्टी और विस्मय शेट्टी जैसे छात्रों द्वारा विकसित किया गया है, और मेघा रानी आर, सहायक प्रोफेसर, एआई और डीएस विभाग, SMVITM के मार्गदर्शन में पूर्ण किया गया है।',
        'input_params': 'इनपुट पैरामीटर',
        'nitrogen': 'नाइट्रोजन (N)',
        'phosphorus': 'फॉस्फोरस (P)',
        'potassium': 'पोटेशियम (K)',
        'temperature': 'तापमान (°C)',
        'humidity': 'नमी (%)',
        'ph': 'पीएच',
        'rainfall': 'वर्षा (मिमी)',
        'Recommend': 'सिफारिश',
        'recommended_crop': 'अनुशंसित फसल',
        'enter_inputs': 'कृपया सभी इनपुट पैरामीटर दर्ज करें।'
    },
    'kn': {
        'title': 'ಕ್ವಿಕ್ ಪ್ರಿಡಿಕ್ಟ್',
        'tagline': 'ಬೆಳೆ ಶಿಫಾರಸು ವ್ಯವಸ್ಥೆ',
        'about': 'ಬಗ್ಗೆ',
        'about_info': 'ಕೃಷಿ ಶಿಫಾರಸು ವ್ಯವಸ್ಥೆ ರೈತರಿಗೆ ಮಣ್ಣು ಮತ್ತು ಹವಾಮಾನ ಪರಿಸ್ಥಿತಿಗಳ ಆಧಾರದ ಮೇಲೆ ಸೂಕ್ತ ಬೆಳೆಯನ್ನು ಆಯ್ಕೆ ಮಾಡಲು ಸಹಾಯ ಮಾಡುತ್ತದೆ. ಈ ವ್ಯವಸ್ಥೆ ಉತ್ತಮ ಬೆಳೆ ನಿರ್ಧರಿಸಲು ಯಂತ್ರ ಕಲಿಕೆಯ ಮಾದರಿಗಳನ್ನು ಬಳಸುತ್ತದೆ. ಇದು (B.E) ಅಭಿಷೇಕ್ ಎ. ಶೆಟ್ಟಿ, ಅನೀಶ್ ಪಿ.ಕೆ, ಹರಿಪ್ರಸಾದ್, ಸಾಗರ್ ಎಸ್. ಶೆಟ್ಟಿ ಮತ್ತು ವಿಸ್ಮಯ ಶೆಟ್ಟಿ ಎಂಬ ವಿದ್ಯಾರ್ಥಿಗಳಿಂದ ಅಭಿವೃದ್ಧಿಪಡಿಸಲಾಗಿದೆ, ಮತ್ತು ಸಲಹೆಗಾರ ಮೇಘಾ ರಾಣಿ ಆರ್, ಸಹಾಯಕರ ಪ್ರಾಧ್ಯಾಪಕರಾದ ಎಐ ಮತ್ತು ಡಿಎಸ್ ವಿಭಾಗ, SMVITM ಅವರ ಮಾರ್ಗದರ್ಶನದಲ್ಲಿ ಸಂಪನ್ನಗೊಂಡಿದೆ',
        'input_params': 'ಇನ್‌ಪುಟ್ ಪ್ಯಾರಾಮೀಟರ್‌ಗಳು',
        'nitrogen': 'ನೈಟ್ರೋಜನ್ (N)',
        'phosphorus': 'ಫಾಸ್ಫರಸ್ (P)',
        'potassium': 'ಪೊಟ್ಯಾಸಿಯಮ್ (K)',
        'temperature': 'ತಾಪಮಾನ (°C)',
        'humidity': 'ಆದ್ರತೆ (%)',
        'ph': 'ಪಿಹೆಚ್',
        'rainfall': 'ಮಳೆ(ಮಿಮೀ)',
        'Recommend': 'ಶಿಫಾರಸು',
        'recommended_crop': 'ಶಿಫಾರಸು ಮಾಡಿದ ಬೆಳೆ',
        'enter_inputs': 'ದಯವಿಟ್ಟು ಎಲ್ಲಾ ಇನ್‌ಪುಟ್ ಪ್ಯಾರಾಮೀಟರ್‌ಗಳನ್ನು ನಮೂದಿಸಿ.'
    }
}

lang = lang_options[selected_lang]
t = translations[lang]

# Streamlit app
st.markdown(f"""
    <h1 style='text-align: center; color: #4CAF50; font-size: 48px; font-family: Arial, sans-serif;'>{t['title']}</h1>
    <h3 style='text-align: center; color: #555555; font-size: 24px; font-family: Arial, sans-serif;'>{t['tagline']}</h3>
    <br><br>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = t['input_params']

if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = pd.DataFrame(columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'prediction'])

if 'result' not in st.session_state:
    st.session_state.result = None

if 'result_crop' not in st.session_state:
    st.session_state.result_crop = None

# Sidebar menu
page = st.sidebar.radio("Select Page", [t['input_params'], t['about']], label_visibility="collapsed")

if page == t['about']:
    st.session_state.page = t['about']
else:
    st.session_state.page = t['input_params']

if st.session_state.page == t['about']:
    st.subheader(t['about'])
    st.info(t['about_info'])
elif st.session_state.page == t['input_params']:
    st.sidebar.header(t['input_params'])

    # Initialize user inputs
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {
            'N': None,
            'P': None,
            'K': None,
            'temperature': None,
            'humidity': None,
            'ph': None,
            'rainfall': None
        }

    def user_input_features():
        N = st.sidebar.number_input(t['nitrogen'], value=st.session_state.user_data['N'], min_value=0.0, step=0.1)
        P = st.sidebar.number_input(t['phosphorus'], value=st.session_state.user_data['P'], min_value=0.0, step=0.1)
        K = st.sidebar.number_input(t['potassium'], value=st.session_state.user_data['K'], min_value=0.0, step=0.1)
        temperature = st.sidebar.number_input(t['temperature'], value=st.session_state.user_data['temperature'], min_value=-50.0, max_value=50.0, step=0.1)
        humidity = st.sidebar.number_input(t['humidity'], value=st.session_state.user_data['humidity'], min_value=0.0, max_value=100.0, step=0.1)
        ph = st.sidebar.number_input(t['ph'], value=st.session_state.user_data['ph'], min_value=0.0, max_value=14.0, step=0.1)
        rainfall = st.sidebar.number_input(t['rainfall'], value=st.session_state.user_data['rainfall'], min_value=0.0, step=0.1)

        data = {
            'N': N,
            'P': P,
            'K': K,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall
        }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    st.subheader(t['input_params'])
    st.write(input_df)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(t['Recommend']):
        # Check if any of the inputs are None
        if any(value is None for value in input_df.iloc[0].values):
            st.warning(t['enter_inputs'])
        else:
            result = {}
            for name, model in models.items():
                pred = model.predict(input_df)
                result[name] = pred[0]

            # Determine the recommended crop based on the Random Forest model's prediction
            recommended_crop = result['Random Forest']
            st.session_state.result = recommended_crop

            crop_images = {
                'rice': 'rice.jpg',
                'maize': 'maize.jpg',
                'chickpea': 'chickpea.jpg',
                'kidneybeans': 'Kidney Beans..jpg',
                'pigeonpeas': 'Pigeon-Peas.jpg',
                'mothbeans': 'moth beans.jpg',
                'mungbean': 'mung beans.jfif',
                'blackgram': 'blackgram.jpg',
                'lentil': 'lentil.jpg',
                'pomegranate': 'Pomegranate.jpg',
                'banana': 'banana.jfif',
                'mango': 'mango.jfif',
                'grapes': 'grapes.jpg',
                'watermelon': 'Watermelon.jpg',
                'muskmelon': 'muskmelon.jpg',
                'apple': 'apple.jfif',
                'orange': 'orange.jpg',
                'papaya': 'papaya.jfif',
                'coconut': 'coconut.jfif',
                'cotton': 'cotton.jfif',
                'jute': 'jute.jfif',
                'coffee': 'coffee.jfif'
            }

            # Get the crop image URL from the mapping
            st.session_state.result_crop = crop_images.get(recommended_crop, 'default.jpg')

            # Store the new prediction
            new_prediction = input_df.copy()
            new_prediction['prediction'] = recommended_crop
            st.session_state.predictions_history = pd.concat([st.session_state.predictions_history, new_prediction], ignore_index=True)

            # Navigate to the result page
            st.session_state.page = 'result'

if st.session_state.page == 'result':
    st.markdown(f"""
        <h2 style='text-align: center; color: #4CAF50; font-size: 36px; font-family: Arial, sans-serif;'>{st.session_state.result}</h2>
    """, unsafe_allow_html=True)
    st.image(st.session_state.result_crop, use_column_width=True)
