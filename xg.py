import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
from sklearn.preprocessing import Imputer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:\\Users\shd\Desktop\\G1.csv")
# print(df)
# 取x，y
train = list(df[['dis', 'age', 'gender']].values)
target = list(df['speed'].values)
# 分割数据集
train_X, test_X, train_y, test_y = train_test_split(train, target, test_size=0.2, random_state=0)
print(len(train_X), len(train_y), len(test_X), len(test_y))

# print(test_X)
#

def trainandTest(X_train, y_train, X_test, y_test):
    # XGBoost训练过程
    b = [x[0] for x in test_X]
    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False, objective='reg:gamma')
    model.fit(X_train, y_train)

    # 对测试集进行预测
    ans = model.predict(X_test)

    ans_len = len(ans)
    id_list = np.arange(0, len(ans))
    data_arr = []
    for row in range(0, ans_len):
        data_arr.append([int(id_list[row]), ans[row]])
    np_data = np.array(data_arr)

    # 写入文件
    pd_data = pd.DataFrame(np_data, columns=['id', 'y'])
    # print(pd_data)
    pd_data.to_csv('C:\\Users\shd\Desktop\\submit.csv', index=None)

    # 显示重要特征
    plot_importance(model)
    plt.show()
    plt.scatter(b, y_test)
    # plt.plot(b, np_data)
    print(len(X_test), len(y_test), len(X_test), len(np_data))
    plt.show()



'''if __name__ == '__main__':
    trainFilePath = 'C:\\Users\shd\Desktop\\dataset/soccer/train.csv'
    testFilePath = 'C:\\Users\shd\Desktop\\dataset/soccer/test.csv'
    data = loadDataset(trainFilePath)
    X_train, y_train = featureSet(data)
    X_test = loadTestData(testFilePath)'''
trainandTest(train_X, train_y, test_X, test_y)
