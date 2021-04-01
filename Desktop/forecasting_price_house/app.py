import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
import streamlit
model=open("random_forest_model.pkl","rb")
RF_model=joblib.load(model)
def RF_prediction(grade,sqft_living,lat,long,sqft_living15,yr_built,waterfront,
                       sqft_above,zipcode,view):
    
    pred_args=[grade,sqft_living,lat,long,sqft_living15,yr_built,waterfront,
                       sqft_above,zipcode,view]
    pred_arr=np.array(pred_args)
    preds=pred_arr.reshape(1,-1)
    #preds=preds.astype(int)
    model_prediction=np.exp(RF_model.predict(preds))-1
    return model_prediction
def run():
    streamlit.title("Random Forest Regression Model")
    html_temp="""
    """
    streamlit.markdown(html_temp)
grade=streamlit.text_input('grade')
sqft_living=streamlit.text_input('sqft_living')
lat=streamlit.text_input('lat')
long=streamlit.text_input('long')
sqft_living15=streamlit.text_input('sqft_living15')
yr_built=streamlit.text_input('yr_built')
waterfront=streamlit.text_input('waterfront')
sqft_above=streamlit.text_input('sqft_above')
zipcode=streamlit.text_input('zipcode')
view=streamlit.text_input('view')    
prediction=" "
if streamlit.button("Predict"):
    prediction=RF_prediction(grade,sqft_living,lat,long,sqft_living15,yr_built,waterfront,
                       sqft_above,zipcode,view)
streamlit.success("The prediction by Model : {}".format(prediction))
if __name__=='__main__':
    run()
