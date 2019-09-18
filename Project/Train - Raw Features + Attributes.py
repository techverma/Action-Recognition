import scipy.io
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import numpy
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV

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

ActionClassifier = joblib.load('Dumps/ActionClassifier.pkl')
AttributeClassifier = joblib.load('Dumps/AttributeClassifier.pkl')

features = joblib.load('Dumps/UIUC_F1.pkl') + joblib.load('Dumps/UIUC_F2.pkl')
labels = joblib.load('Dumps/UIUC_L1.pkl') + joblib.load('Dumps/UIUC_L2.pkl')

TRAIN_FEATURES=[]
TRAIN_LABELS=[]
TEST_FEATURES=[]
TEST_LABELS=[]
for feature,label in zip(features,labels):
    FEATURE_VECTOR=[]
    action = ActionClassifier.decision_function(numpy.array(feature).reshape(1, -1))
    attributes = AttributeClassifier.decision_function(numpy.array(feature).reshape(1, -1))
    TRAIN_LABELS.append(label)
    for i in action[0]:
        FEATURE_VECTOR.append(i)
    for i in attributes[0]:
        FEATURE_VECTOR.append(i)
    for i in range(len(attributes[0])):
        for j in range(i+1,len(attributes[0])):
            if attributes[0][i]==1:
                if attributes[0][j]==1:
                    FEATURE_VECTOR.extend((0,0,0,1))
                else:
                    FEATURE_VECTOR.extend((0,0,1,0))
            else:
                if attributes[0][j]==1:
                    FEATURE_VECTOR.extend((0,1,0,0))
                else:
                    FEATURE_VECTOR.extend((1,0,0,0))
    TRAIN_FEATURES.append(FEATURE_VECTOR)

features = joblib.load('Dumps/UIUC_F3.pkl')
labels = joblib.load('Dumps/UIUC_L3.pkl')

for feature,label in zip(features,labels):
    FEATURE_VECTOR=[]
    action = ActionClassifier.decision_function(numpy.array(feature).reshape(1, -1))
    attributes = AttributeClassifier.decision_function(numpy.array(feature).reshape(1, -1))
    TEST_LABELS.append(label)
    for i in action[0]:
        FEATURE_VECTOR.append(i)
    for i in attributes[0]:
        FEATURE_VECTOR.append(i)
    for i in range(len(attributes[0])):
        for j in range(i+1,len(attributes[0])):
            if attributes[0][i]>0:
                if attributes[0][j]>0:
                    FEATURE_VECTOR.extend((0,0,0,1))
                else:
                    FEATURE_VECTOR.extend((0,0,1,0))
            else:
                if attributes[0][j]>0:
                    FEATURE_VECTOR.extend((0,1,0,0))
                else:
                    FEATURE_VECTOR.extend((1,0,0,0))
    TEST_FEATURES.append(FEATURE_VECTOR)

classifier = LinearSVC(C=1.0, random_state=0)

TRAIN_FEATURES= numpy.array(TRAIN_FEATURES)
TRAIN_LABELS = numpy.array(TRAIN_LABELS)
TEST_FEATURES= numpy.array(TEST_FEATURES)
TEST_LABELS = numpy.array(TEST_LABELS)
classifier.fit(TRAIN_FEATURES,TRAIN_LABELS)
prediction = classifier.predict(TEST_FEATURES)
print accuracy_score(TEST_LABELS,prediction)
joblib.dump(classifier,'Dumps/LatentClassifierRaw+Attributes.pkl')
