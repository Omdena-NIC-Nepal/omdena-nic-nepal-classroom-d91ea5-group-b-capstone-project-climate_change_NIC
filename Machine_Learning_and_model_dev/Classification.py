import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


df = pd.read_csv("../Weather_&_Climate_Data/Processed Datas/Feature_Engineering/climate_df_scaled.csv")

def climate_zone_cluster_classify():
    
    X = df[["Temp_2m_scaled", 'Humidity_2m_scaled', 'Pressure_scaled', 'WindSpeed_10m_scaled']]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters= 3, random_state= 42)
    df['Climate_Zone'] = kmeans.fit_predict(X_scaled)
    #print this line in seaborn print(df.groupby('Climate_Zone').mean())

    X_train, X_test, y_train, y_test = train_test_split(X, df['Climate_Zone'], test_size=0.3, random_state= 42)
    X_test = scaler.transform(X_test)

    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
#print this line in seaborn     print(f'Accuracy: {accuracy * 100:.2f}%')

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_labels = y_test.unique()
    # Get unique climate zones (e.g., ["Hot/Dry", "Moderate", "Cold/Humid"])

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False, 
                xticklabels=class_labels, yticklabels=class_labels)

    plt.title('Confusion Matrix (Climate Zone Classification)')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
   
    
    plt.show()
        
climate_zone_cluster_classify()

def SVM_classifier():
    threshold = np.percentile(df['Temp_2m_scaled'], 95)
    df['Extreme_Event'] = (
    (df['Temp_2m_scaled'] >= threshold) & 
    (df['Humidity_2m_scaled'] < np.percentile(df['Humidity_2m_scaled'], 5)) & 
    (df['WindSpeed_10m_scaled'] > np.percentile(df['WindSpeed_10m_scaled'], 95))
    ).astype(int)
    print(df['Extreme_Event'].value_counts())

    X = df[['Temp_2m_scaled', 'Humidity_2m_scaled', 'Pressure_scaled', 'WindSpeed_10m_scaled']]
    y = df['Extreme_Event']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))    

SVM_classifier()

def gradient_boosting_assessment():
    # df['Temp_Extreme'] = (df['Temp_2m_scaled'] > np.percentile(df['Temp_2m_scaled'], 95)). astype(int)
    # df['Humidity_Extreme'] = (df['Humidity_2m_scaled'] < np.percentile(df['Humidity_2m_scaled'], 5)).astype(int)
    # df['Wind_Extreme'] = (df['WindSpeed_10m'] > np.percentile(df['WindSpeed_10m_scaled'], 95)).astype(int)

    # df['Vulnerability_Score'] = df[['Temp_Extreme', 'Humididity_Extreme', 'Wind_Extreme']].sum(axis=1)*33.33
    from sklearn.cluster import KMeans
    X = df[['Temp_2m_scaled', 'Humidity_2m_scaled', 'WindSpeed_10m_scaled']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Risk_Level'] = kmeans.fit_predict(X)  # 0=Low, 1=Medium, 2=High (interpret post-hoc)

    y = df['Risk_Level']  # From Option 2

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100)  # Use 'multi:softmax' for >2 classes
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
    print(f"Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.2f}")

gradient_boosting_assessment()