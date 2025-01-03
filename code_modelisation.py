#region import du fichier, préprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

df = pd.read_csv('data_cleaned.csv', index_col=0)
X = df.drop(columns='Ewltp (g/km)')
y = df['Ewltp (g/km)'] 

#Différenciation des colonnes numériques et catégorielles 
num_col_NA = ['m (kg)','ec (cm3)','ep (KW)','Fuel consumption ']
num_col = ['m (kg)','ec (cm3)','ep (KW)','Fuel consumption ','Electric range (km)','Erwltp (g/km)']
cat_col = ['Ct','fuel_type']

     
#Gestion des valeurs manquantes
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
X.loc[:,num_col_NA] = imputer.fit_transform(X[num_col_NA])

#Encodage des variables catégorielles
oneh = OneHotEncoder(drop = 'first', sparse_output=False)
X_encoded = oneh.fit_transform(X[cat_col])

#Conversion en DataFrame
noms_colonnes_cat = oneh.get_feature_names_out(cat_col)
X_encoded = pd.DataFrame(X_encoded, columns=noms_colonnes_cat, index = X.index)

#Standardisation des variables numériques
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[num_col])

#Conversion en DataFrame
noms_colonnes_num = scaler.get_feature_names_out(num_col)
X_scaled = pd.DataFrame(X_scaled, columns=noms_colonnes_num, index = X.index)

#Reconstition du tableau après encodage
X.drop(columns=cat_col)
X = pd.concat([X_encoded,X_scaled], axis = 1)
#endregion

#region entrainement modélisation
rf = RandomForestRegressor()
rf.fit(X, y)

import joblib
joblib.dump(rf, "model.pkl")
#endregion