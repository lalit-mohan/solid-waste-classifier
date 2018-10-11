def main():
    import numpy as np
    import pandas as pd
    import cv2
    from scipy import misc, ndimage
    import matplotlib.pyplot as plt
    from xgboost import XGBClassifier
    from sklearn.model_selection import GridSearchCV

# def grab_data():
# 	dataList = []
# 	for i in range(1,483):
# 	    img = cv2.imread('dataset\\plastic\\plastic'+str(i)+'.jpg')
# 	    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 	    img_new = misc.imresize(img, (28, 28))
# 	    hist = cv2.calcHist([img_new],[0],None,[256],[0,256])
# 	    dataList.append(hist)
# 	    if i==482:
# 	        plt.hist(hist)
# 	        plt.show()
# 	items = ['metal','glass', 'cardboard', 'paper']
# 	for x in items:
# 	    for i in range(1,401):
# 	        img = cv2.imread('dataset\\'+x+'\\'+x+str(i)+'.jpg')
# 	        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 	        img_new = misc.imresize(img, (28, 28))
# 	        hist = cv2.calcHist([img_new],[0],None,[256],[0,256])
# 	        dataList.append(hist)

# 	k = np.array(dataList)
# 	k = k.reshape(2082,256)
# 	df1 = pd.DataFrame(k)
# 	df1['labels'] = pd.Series(np.zeros(2082), index=df1.index)
# 	df1.iloc[:482,-1]='plastic'
# 	df1.iloc[482:882,-1]='metal'
# 	df1.iloc[882:1282,-1]='glass'
# 	df1.iloc[1282:1682,-1]='cardboard'
# 	df1.iloc[1682:2082,-1]='paper'
# 	df1 = df1.sample(frac=1).reset_index(drop=True)
# 	df1.to_csv('dataFrame.csv',index=None)
# 	return df1

# def train_clf(df1):
# 	X = df1.iloc[:,:-1].values
# 	Y = df1.iloc[:,-1:].values
# 	# reg = XGBClassifier()
# 	# parameters = [{ 'max_depth': [5], 'min_child_weight': [4], 'subsample': [0.9],'n_estimators': [110]}]

# 	# grid_search = GridSearchCV(estimator = reg, param_grid = parameters, scoring = 'accuracy', cv = 5, n_jobs = -1, verbose = 2)
# 	# grid_search = grid_search.fit(X, Y)

# 	# best_acc = grid_search.best_score_
# 	# best_param = grid_search.best_params_
# 	# return grid_search
# 	clc5 = XGBClassifier(max_depth=5, min_child_weight = 3, subsample= 0.9 , n_estimators = 110)
# 	clc5.fit(X, Y)
# 	return clc5


    df1 = pd.read_csv('dataFrame.csv')
    X = df1.iloc[:,:-1].values
    Y = df1.iloc[:,-1:].values
    clf = XGBClassifier(max_depth=5, min_child_weight = 3, subsample= 0.9 , n_estimators = 110)
    clf.fit(X, Y)
    return clf

if __name__ == '__main__':
	main()

