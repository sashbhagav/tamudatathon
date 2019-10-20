import sys
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, redirect, Response
import random, json
import pandas as pd

# final = pd.read_csv('sendtoml.csv')
# # Elbow method to determine the number of K in Kmeans Clustering
# def kmeans(final):
#     coords = final[['Longitude', 'Latitude']]

#     distortions = []
#     K = range(1,25)
#     for k in K:
#         kmeansModel = KMeans(n_clusters=k)
#         kmeansModel = kmeansModel.fit(coords)
#         distortions.append(kmeansModel.inertia_)
        
#     fig, ax = plt.subplots(figsize=(12, 8))
#     plt.plot(K, distortions, marker='o')
#     plt.xlabel('k')
#     plt.ylabel('Distortions')
#     plt.title('Elbow Method For Optimal k')
#     plt.savefig('elbow.png')
#     plt.show()


#     sil = []
#     kmax = 50

#     # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
#     for k in range(2, kmax+1):
#       kmeans = KMeans(n_clusters = k).fit(coords)
#       labels = kmeans.labels_
#       sil.append(silhouette_score(coords, labels, metric = 'euclidean'))

#     kmeans = KMeans(n_clusters=5, init='k-means++')
#     kmeans.fit(coords)
#     y = kmeans.labels_
#     print("k = 5", " silhouette_score ", silhouette_score(coords, y, metric='euclidean'))

#     final['cluster'] = kmeans.predict(final[['Longitude','Latitude']])

# def recommend_restaurants(df, longitude, latitude):
#     # Predict the cluster for longitude and latitude provided
#     cluster = kmeans.predict(np.array([longitude,latitude]).reshape(1,-1))[0]
#     print(cluster)
   
#     # Get the best restaurant in this cluster
#     return  df[df['cluster']==cluster].iloc[0:10][['postalCode', 'Latitude','Longitude']]

app = Flask(__name__)
@app.route('/receivedata', methods=['POST'])
def receive_data():
    print request.form['myData']

# @app.route('/') 

# def foo():
#     # data = request.json
#     return "hello"
    # print(data[0])
    # kmeans(final)
    # recommend_restaurants(final, data[0], data[1])
    # return jsonify(data)