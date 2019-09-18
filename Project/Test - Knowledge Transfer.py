import scipy.io
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import numpy
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import DistanceMetric

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

AttributeClassifier = joblib.load('Dumps/AttributeClassifierKnowledgeTransfer.pkl')

features = scipy.io.loadmat("./UIUC1/UIUC1_win_feature.mat")
labels = scipy.io.loadmat("./UIUC1/UIUC1_labels.mat")
action_actor = open("./UIUC1/action_actor.txt")

dist = DistanceMetric.get_metric('euclidean')

mapping = [{}]
for line in action_actor:
    line = line.split()
    actionvector = numpy.zeros(14, dtype=numpy.int)
    actionvector[int(line[0])]=1
    mapping.append({'action':int(line[0]),'actionvector':actionvector, 'actor':int(line[1])})

total = len(labels['vlabels'][0])
ConfusionMatrix=numpy.array([[0,0],[0,0]])
NovelClassList=[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13]]
for NovelClass in NovelClassList:

    ConfusionMatrix2=numpy.array([[0,0],[0,0]])

    TRAIN_LABELS = []
    TRAIN_FEATURES = []
    TRAIN_ATTRIBUTE = []

    TEST_LABELS = []
    TEST_FEATURES = []
    TEST_ATTRIBUTE = []

    for feature,label in zip(features['win_feature'],labels['vlabels'][0]):
        if mapping[int(label)]['action'] not in NovelClass:
            # if label<463:
            TRAIN_LABELS.append(mapping[int(label)]['action'])
            TRAIN_FEATURES.append(feature)
            TRAIN_ATTRIBUTE.append(attribute_mapping[int(mapping[int(label)]['action'])])
            # else:
            #     TEST_LABELS.append(mapping[int(label)]['action'])
            #     TEST_FEATURES.append(feature)
            #     TEST_ATTRIBUTE.append(attribute_mapping[int(mapping[int(label)]['action'])])
        # if lineno==4000:
        #     break

    TRAIN_FEATURES = numpy.array(TRAIN_FEATURES)
    # TEST_FEATURES = numpy.array(TEST_FEATURES)
    TEST_FEATURES = TRAIN_FEATURES

    TRAIN_LABELS = numpy.array(TRAIN_LABELS)
    # TEST_LABELS = numpy.array(TEST_LABELS)
    TEST_LABELS = TRAIN_LABELS

    TRAIN_ATTRIBUTE = numpy.array(TRAIN_ATTRIBUTE)
    # TEST_ATTRIBUTE = numpy.array(TEST_ATTRIBUTE)
    TEST_ATTRIBUTE = TRAIN_ATTRIBUTE

    pc=0
    nc=0
    classifier = OneVsRestClassifier(LinearSVC(C=2.0,random_state=0))
    classifier.fit(TRAIN_FEATURES,TRAIN_ATTRIBUTE)
    decision = classifier.decision_function(TEST_FEATURES)
    prediction = classifier.predict(TEST_FEATURES)
    for i in range(0,len(TEST_ATTRIBUTE)):
        for j in range(22):
            if prediction[i][j]==TEST_ATTRIBUTE[i][j]:
                pc+=1
            else:
                nc+=1
        # print prediction[i],TEST_ATTRIBUTE[i],TEST_LABELS[i], decision[i]
    print pc,nc
    print classifier.score(TEST_FEATURES,TEST_ATTRIBUTE)

    TRAIN_LABELS = []
    TRAIN_FEATURES = []
    TRAIN_ATTRIBUTE = []

    TEST_LABELS = []
    TEST_FEATURES = []
    TEST_ATTRIBUTE = []

    for feature,label in zip(features['win_feature'],labels['vlabels'][0]):
        if mapping[int(label)]['action'] in NovelClass:
            attributes = classifier.predict(feature.reshape(1, -1))
            TEST_LABELS.append(mapping[int(label)]['action'])
            TEST_FEATURES.append(numpy.array(attributes[0]))
            A = dist.pairwise([attributes[0],numpy.array(attribute_mapping[NovelClass[0]])])[0][1]
            B = dist.pairwise([attributes[0],numpy.array(attribute_mapping[NovelClass[1]])])[0][1]
            if A>B:
                ConfusionMatrix[1][mapping[int(label)]['action']-NovelClass[0]]+=1
                ConfusionMatrix2[1][mapping[int(label)]['action']-NovelClass[0]]+=1
            else:
                ConfusionMatrix[0][mapping[int(label)]['action']-NovelClass[0]]+=1
                ConfusionMatrix2[0][mapping[int(label)]['action']-NovelClass[0]]+=1

    print ConfusionMatrix2
    print 'Accuracy:'(1.0*numpy.trace(ConfusionMatrix2))/numpy.sum(ConfusionMatrix2)
print ConfusionMatrix
print 'Accuracy:'(1.0*numpy.trace(ConfusionMatrix))/numpy.sum(ConfusionMatrix)
