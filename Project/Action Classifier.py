import scipy.io
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import numpy
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
import operator

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

action_name = [
  "hand-clap",
  "crawl",
  "jump forward",
  "jump from situp",
  "jumping jacks",
  "pushing up",
  "raise 1 hand",
  "run",
  "sitting to standing",
  "standing to sitting",
  "stretch out",
  "turn",
  "walking",
  "waving"
]

ActionClassifier = joblib.load('Dumps/ActionClassifier.pkl')
AttributeClassifier = joblib.load('Dumps/AttributeClassifier.pkl')
DataDrivenAttributes = joblib.load('Dumps/DataDrivenAttributes.pkl')
classifier=joblib.load('Dumps/LatentClassifierRaw+Attributes+DataDriven.pkl')

while(True):
    print "Video ID:"
    filename = raw_input()

    features = joblib.load('Videos/'+str(filename)+'.pkl')
    label = features['label']
    features = features['feature']

    predictions={}

    for feature in features:
        FEATURE_VECTOR=[]
        action = ActionClassifier.decision_function(numpy.array(feature).reshape(1, -1))
        attributes = AttributeClassifier.decision_function(numpy.array(feature).reshape(1, -1))
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
        scoreValue = DataDrivenAttributes.transform(feature.reshape(1, -1))
        for i in scoreValue[0]:
            FEATURE_VECTOR.append(0.25/(0.25+i))
        for i in range(len(scoreValue[0])):
            for j in range(i+1,len(scoreValue[0])):
                if scoreValue[0][i]>0.25:
                    if scoreValue[0][j]>0.25:
                        FEATURE_VECTOR.extend((0,0,0,1))
                    else:
                        FEATURE_VECTOR.extend((0,0,1,0))
                else:
                    if scoreValue[0][j]>0.25:
                        FEATURE_VECTOR.extend((0,1,0,0))
                    else:
                        FEATURE_VECTOR.extend((1,0,0,0))
        prediction=classifier.predict(FEATURE_VECTOR)[0]
        decision=classifier.decision_function(FEATURE_VECTOR)[0][prediction]
        if prediction not in predictions:
            predictions[prediction]=0.0
        predictions[prediction]+=1000000+decision

    print 'True Label: ', action_name[label]
    print 'Predicted Label: ', action_name[max(predictions.iteritems(), key=operator.itemgetter(1))[0]]
