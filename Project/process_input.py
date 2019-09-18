import scipy.io
from sklearn.externals import joblib

features = scipy.io.loadmat("./UIUC1/UIUC1_win_feature.mat")
labels = scipy.io.loadmat("./UIUC1/UIUC1_labels.mat")
action_actor = open("./UIUC1/action_actor.txt")
output = open("UIUC1_formated.txt","w")

mapping = [{}]
for line in action_actor:
    line = line.split()
    mapping.append({'action':int(line[0]), 'actor':int(line[1])})

total = len(labels['vlabels'][0])
output.write('%s\n' % total)

action_decision = joblib.load('Dumps/ActionDecisionValuesTraining.pkl')
attribute_decision = joblib.load('Dumps/AttributeDecisionValuesTraining.pkl')

assert(len(features['win_feature'])==len(action_decision))
assert(len(action_decision)==len(attribute_decision))

lineno=0
for feature,label in zip(features['win_feature'],labels['vlabels'][0]):
    # if lineno>40000:
    #     break
    for value in feature:
        output.write(str(value)+' ')
    for value in action_decision[lineno]:
        output.write(str(value)+' ')
    for value in attribute_decision[lineno]:
        output.write(str(value)+' ')
    output.write(str(mapping[label]['action'])+' ')
    output.write(str(lineno)+'\n')
    lineno+=1
output.close()
