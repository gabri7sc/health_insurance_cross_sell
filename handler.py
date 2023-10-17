import pickle
import pandas as pd
from flask import Flask, request, Response
from healthInsurance.HealthInsurance import HealthInsurance
import os

# loading model
model = pickle.load( open( 'model/model_linear_regression.pkl', 'rb' ) )

# initialize API
app = Flask(__name__)

@app.route( '/healthinsurance/predict', methods=['POST'] )
def healthinsurance_predict():
    test_json = request.get_json()
   
    if test_json: # there is data
        if isinstance( test_json, dict ): # unique example
            test_raw = pd.DataFrame( test_json, index=[0] )
            
        else: # multiple example
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )
            
        pipeline = HealthInsurance()

        #Data Description
        df1 = pipeline.rename_columns(test_raw)

        #Data Transformation
        df2 = pipeline.rename_categorical(df1)

        #Data Encoding
        df3 = pipeline.data_encoding(df2)

        #Data Rescalling
        df4 = pipeline.data_rescalling(df3)

        #prediction
        df_response = pipeline.get_prediction(model, test_raw, df4)
        
        return df_response
        
        
    else:
        return Response( '{}', status=200, mimetype='application/json' )
    
#Usar isso para testar na maquina
#if __name__ == '__main__':
    #app.run('0.0.0.0', debug=True)

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)
