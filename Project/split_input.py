import scipy.io
from sklearn.externals import joblib

features = scipy.io.loadmat("./UIUC1/UIUC1_win_feature.mat")
labels = scipy.io.loadmat("./UIUC1/UIUC1_labels.mat")
action_actor = open("./UIUC1/action_actor.txt")

sec1=[]
sec2=[]
sec3=[]
sec4=[]
sec5=[]
sec6=[]

mapping = [{}]
for line in action_actor:
    line = line.split()
    mapping.append({'action':int(line[0]), 'actor':int(line[1])})

for feature,label in zip(features['win_feature'],labels['vlabels'][0]):
    if label<272:
        sec1.append(feature)
        sec4.append(mapping[label]['action'])
    elif label<463:
        sec2.append(feature)
        sec5.append(mapping[label]['action'])
    else:
        sec3.append(feature)
        sec6.append(mapping[label]['action'])

joblib.dump(sec1,'Dumps/UIUC_F1.pkl')
joblib.dump(sec2,'Dumps/UIUC_F2.pkl')
joblib.dump(sec3,'Dumps/UIUC_F3.pkl')
joblib.dump(sec4,'Dumps/UIUC_L1.pkl')
joblib.dump(sec5,'Dumps/UIUC_L2.pkl')
joblib.dump(sec6,'Dumps/UIUC_L3.pkl')
