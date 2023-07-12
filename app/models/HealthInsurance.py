import pickle
import numpy  as np
import pandas as pd


class HealthInsurance:
    
    def __init__( self ):
        self.home_path =''
        self.gender_encoding = pickle.load( open( '/features/gender_encoding.pkl', 'rb' ) )
        self.vehicle_damage_encoding = pickle.load( open( self.home_path + '\features\vehicle_damage_encoding.pkl', 'rb' ) ) 
        self.vehicle_age_encoding = pickle.load( open( self.home_path + '\features\vehicle_age_encoding.pkl', 'rb' ) )
        self.region_code_encoding = pickle.load( open( self.home_path + '\features\region_code_encoding.pkl', 'rb' ) )
        self.policy_sales_channel_encoding = pickle.load( open( self.home_path + '\features\policy_sales_channel_encoding.pkl', 'rb' ) )
        self.age_scaler = pickle.load( open( self.home_path + '\features\age_scaler.pkl', 'rb' ) )
        self.annual_premium_scaler = pickle.load( open( self.home_path + '\features\annual_premium_scaler.pkl', 'rb' ) )
        self.vintage_scaler = pickle.load( open( self.home_path + '\features\vintage_scaler.pkl', 'rb' ) )
        
        
    def data_cleaning (self, df2):
        
        # Renomeação de colunas
        
        new_columns_name = ['id', 'gender', 'age', 'driving_license', 'region_code', 'previously_insured', 'vehicle_age', 
            'vehicle_damage', 'annual_premium', 'policy_sales_channel', 'vintage']

        df2.columns = new_columns_name
        
        
        return df2
    
    
    
    def data_preparation (self, df_train):
        
        # Codificações
        
        df_train['gender'] =  df_train['gender'].map(self.gender_encoding)
        
        df_train['vehicle_damage'] =  df_train['vehicle_damage'].map(self.vehicle_damage_encoding)

        df_train["vehicle_age"] = self.vehicle_age_encoding.transform(df_train[["vehicle_age"]])
        
        df_train['region_code'] =  df_train['region_code'].map(self.region_code_encoding)
        
        df_train['policy_sales_channel'] =  df_train['policy_sales_channel'].map(self.policy_sales_channel_encoding)
        
        
        # Reescalas
        
        df_train["age"] = self.age_scaler.transform(df_train[["age"]])
        
        df_train["annual_premium"] = self.annual_premium_scaler.transform(df_train[["annual_premium"]])
        
        df_train["vintage"] = self.vintage_scaler.transform(df_train[["vintage"]])
        
        
        # Seleção de features

        features_selected = ['previously_insured', 'vehicle_damage', 'vehicle_age', 'age', 'annual_premium', 'vintage', 'policy_sales_channel']
        
        
        return df_train[features_selected]
        
        

    def get_prediction (self, model, original_data, test_data):
        
        # Predição do modelo
        
        pred = model.predict_proba(test_data)[:,1]
        
        
        # Junção da predição aos dados originais
        
        original_data['score'] = pred
        
        
        return original_data.to_json( orient='records', date_format='iso' )
