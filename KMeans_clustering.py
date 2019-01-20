from matplotlib import pyplot as pt 
import numpy as np
from sklearn.cluster import KMea
import pandas as pd
df = pd.read_csv("E:/college/Data_KMeans.csv", sep=',')

f1 = df['Distance_Feature'].values
f2 = df['Speeding_Feature'].values

X=np.array(list(zip(f1,f2)))

sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X)
    df["clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
pt.figure()
pt.plot(list(sse.keys()), list(sse.values()))
pt.title('Elbow Graph')
pt.xlabel("Number of cluster")
pt.ylabel("Sum Of Squared Errors")
pt.show()
