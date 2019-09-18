import scipy.io
from sklearn.externals import joblib

features = scipy.io.loadmat("./UIUC1/UIUC1_win_feature.mat")
labels = scipy.io.loadmat("./UIUC1/UIUC1_labels.mat")
action_actor = open("./UIUC1/action_actor.txt")

Videos={}

mapping = [{}]
for line in action_actor:
    line = line.split()
    mapping.append({'action':int(line[0]), 'actor':int(line[1])})

for feature,label in zip(features['win_feature'],labels['vlabels'][0]):
    if label not in Videos:
        Videos[label]={'label':mapping[label]['action'],'feature':[]}
    Videos[label]['feature'].append(feature)

# print Videos
for video in Videos:
    joblib.dump(Videos[video],'Videos/'+str(video)+'.pkl')
