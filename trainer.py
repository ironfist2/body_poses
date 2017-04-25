from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import pickle
f = open('training2.txt','r')
x = []
y = []
for l in f:
	temp = l.split(',')
	print temp
	x_temp = []
	x_temp.append(1)
	x_temp.append(float(temp[0]))
	x_temp.append(float(temp[1]))
	x_temp.append(float(temp[2]))
	x_temp.append(float(temp[3]))
	x.append(x_temp)
	y.append(int(temp[4]))
log = tree.DecisionTreeClassifier()
log.fit(x,y) 
print log.predict([[1,90,90,8100,8100]])
p = open('train_model.pickle','wb')
t = {'log':log}
pickle.dump(t,p)

# training_file = open('training.txt','r')
# training_file2 = open('training2.txt','w')
# for l in training_file:
# 	temp = l.split(',')
# 	theta3 = float(temp[0])**2
# 	theta4 = float(temp[1])**2
# 	y = int(temp[2])
# 	temp[2] = str(theta3)
# 	temp.append(str(theta4))
# 	temp.append(str(y))
# 	te = ''
# 	for j in range(5):
# 		te += temp[j] + ','
# 	te += '\n'
# 	training_file2.write(te)