import os
import pickle
import pandas as pd
from flask import Flask, request, Response
from models.HealthInsurance import HealthInsurance


# Carregamento do modelo

model = pickle.load( open( 'models/xgb_model.pkl', 'rb') )


# Inicialização da API

app = Flask( __name__ )

@app.route( '/predict', methods=['POST'] )

def health_insurance_predict():
    
    test_json = request.get_json()
    
    if test_json: # there is data
        if isinstance( test_json, dict ): # unique example
            test_raw = pd.DataFrame( test_json, index=[0] )
            
        else: # multiple example
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )
            
        # Instanciamento da classe
        pipeline = HealthInsurance()
        
        # Limpeza dos dados
        df1 = pipeline.data_cleaning(test_raw)
        
        # Preparação dos dados
        df2 = pipeline.data_preparation(df1)
        
        # Predição do modelo
        df_response = pipeline.get_prediction( model, test_raw, df2 )
        
        return df_response
    
    else:
        return Response( '{}', status=200, mimetype='application/json' )
    
if __name__ == '__main__':
    port = os.environ.get ('PORT', 5000)
    app.run( '0.0.0.0', debug=True )
