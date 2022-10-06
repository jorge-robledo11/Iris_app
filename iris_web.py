# Librerías
import streamlit as st
import pickle
import pandas as pd

# Extraer los archivos pkl
with open('LinearRegression.pkl', 'rb') as file:
    lr = pickle.load(file)
    
with open('LogisticRegression.pkl', 'rb') as file:
    logit = pickle.load(file)
    
with open('SVC.pkl', 'rb') as file:
    svc = pickle.load(file)
    

# Función de predicción
def classification(pred):
    
    if pred == 0:
        return 'Setosa'

    elif pred == 1:
        return 'Versicolor'
    
    else:
        return 'Virginica'
    

# Entry point
# Para correr el script: streamlit run «file.py»
def main():
    
    st.title('Modelamiento de Iris')
    st.sidebar.header('User Input Parameters')
    
    # Inputs suministrados por el usuario
    def user_inputs():
        
        sepal_length = st.sidebar.slider(label='Sepal length', min_value=4.3, max_value=7.9, value=5.4)
        sepal_width = st.sidebar.slider(label='Sepal width', min_value=2.0, max_value=4.4, value=3.4)
        petal_length = st.sidebar.slider(label='Petal length', min_value=1.0, max_value=6.9, value=1.3)
        petal_width = st.sidebar.slider(label='Petal width', min_value=0.1, max_value=2.5, value=0.2)
        
        data = {
            'Sepal Length': sepal_length,
            'Sepal Width': sepal_width,
            'Petal Length': petal_length,
            'Petal Width': petal_width,
        }
        features = pd.DataFrame(data, index=[0])
        return features
    
    # Datos del usuario transformados a Dataframe
    data = user_inputs()
    # Elegir el modelo
    option = ['Linear Regression', 'Logistic Regression', 'Support Vector Classifier']
    model = st.sidebar.selectbox('Which model do you like to use?', option)
    
    st.subheader('Inputs by user')
    st.subheader(model)
    st.write(data)
    
    if st.button('Run'):
        
        if model == option[0]:
            st.success(classification(lr.predict(data)))
        
        elif model == option[1]:
            st.success(classification(logit.predict(data)))
            
        else:
            st.success(classification(svc.predict(data)))

if __name__ == '__main__':
    main()
