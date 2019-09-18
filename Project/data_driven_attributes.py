import scipy.io
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import numpy
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


attribute_mapping = [
  [1,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
  [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0],
  [0,0,1,1,0,1,0,1,1,1,1,1,0,1,1,0,0,1,1,1,1,1],
  [1,1,0,1,0,0,0,0,1,1,0,0,1,0,0,1,1,0,0,0,0,1],
  [1,1,0,1,1,0,1,0,1,1,0,0,0,0,1,0,0,0,0,0,1,1],
  [0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,1,0,0,0,1,0],
  [1,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
  [0,0,1,1,0,1,0,1,1,0,1,1,0,1,1,0,0,1,1,1,1,1],
  [1,1,0,0,0,0,0,0,0,1,0,0,1,0,0,1,1,0,0,0,0,0],
  [1,1,0,0,1,0,0,0,0,1,0,0,1,0,0,1,1,0,0,0,0,0],
  [1,1,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
  [1,1,0,0,0,0,0,0,0,1,0,0,1,0,0,1,1,0,0,0,0,0],
  [0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,0,0,1,1,1,1,0],
  [1,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  ]

features = numpy.array(joblib.load('Dumps/UIUC_F1.pkl') + joblib.load('Dumps/UIUC_F2.pkl'))
labels_true = numpy.array(joblib.load('Dumps/UIUC_L1.pkl') + joblib.load('Dumps/UIUC_L2.pkl'))
features = StandardScaler().fit_transform(features)

db = KMeans(n_clusters=22)
db.fit(features)
# core_samples_mask = numpy.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
clusters = db.cluster_centers_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f"
#       % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print clusters
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(features, labels))
if n_clusters_ > 1:
    joblib.dump(db,'Dumps/DataDrivenAttributes.pkl')
