#!/usr/bin/env python
# coding: utf-8

# # 사이킷런 공부

# In[1]:


pip install scikit-learn


# In[2]:


import sklearn


# In[3]:


print(sklearn.__version__)


# In[4]:


#붓꽃품종 예측하기
#Classification
from sklearn.datasets import load_iris #사이킷런에서 자체적으로 제공하는 데이터 셋
from sklearn.tree import DecisionTreeClassifier #트리기반 ML알고리즘 구현
from sklearn.model_selection import train_test_split #다양한 모듈의 모임


# In[5]:


import pandas as pd


# In[6]:


#붓꽃 데이터 세트 로딩
iris= load_iris()

#iris.data는 Iris 데이터 세트에서 feature만으로 된 데이터를 numpy로 가지고있다.
iris_data=iris.data

iris_label= iris.target
print('iris target 값:',iris_label)
print('iris target 명',iris.target_names)

#dataframe 으로 변환
iris_df=pd.DataFrame(data=iris_data,columns=iris.feature_names)
iris_df['label']=iris.target
iris_df.head(3)


# In[7]:


##학습용 데이터와 테스트용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label,test_size=0.2,random_state=11)


# In[8]:


#DesicionTreeClassifier 객체 생성
df_clf=DecisionTreeClassifier(random_state=11)

#학습 수행
df_clf.fit(X_train,y_train)


# In[9]:


#학습이 완료된 DecisionTreeClassifier 객체에서 테스트 데이터 세트로 예측 수행
pred = df_clf.predict(X_test)


# In[10]:


from sklearn.metrics import accuracy_score
print("예측 정확도:{0:.4f}".format(accuracy_score(y_test,pred)))


# ### 1.데이터 세트 분리: 데이터를 학습 데이터와 테스트 데이터로 분리
# ### 2.모델 학습: 학습 데이터를 기반으로 ML알고리즘을 적용해 모델을 학습
# ### 3. 예측 수행: 학습된 ML 모델을 이용해 테스트 데이터의 분류(불꽃 종류) 를 예측
# ### 4. 평가: 이렇게 예측된 결과값과 테스트 데이터의 실제 결과값을 비교해 ML모델 성능 평가

# # Estimator 이해 및 fit() , predict() 메서드
# - 사이킷런은 매우 많은 유형의 Classifier와 Regressor를 제공하는데 이걸 합쳐서 Estimator클래스라고 부름
# 

# ## 분류나 회귀 연습용 예제 데이터
# - datasets.load_boston() :회귀용도, 보스턴의 집 피처들과 가격에 대한 데이터셋
# - datasets.load_breast_cancer() :분류 용도이며, 위스콘신 유방암 피처들과 악성/음성 레이블 데이터 세트
# - datasets.load_diabetes() : 회귀용도이며,당뇨데이터 세트
# - datasets.load_digits() : 분류 용도이며,0에서 9까지의 숫자의 이미지 픽셀 데이터셋
# - datasets.load_digits() : 분류용도이며, 불꽃에 대한 피처를 가진데이터 셋
# 

# ## 분류와 클러스터링을 위한 표본 데이터 생성기
# - datasets.make_classifications() : 분류를위한 무작위 데이터셋을 만든다. 높은 상관도, 불필요한 속성 등의 노이즈 효과를 위한 데이터
# - datasets.make_blobs() :클러스터링을 위한 데이터셋을 무작위 생성. 

# In[11]:


from sklearn.datasets import load_iris


# In[12]:


iris_data = load_iris()


# In[13]:


print(type(iris_data))


# In[14]:


keys=iris_data.keys()
print('불꽃 데이터 세트의 키들:',keys)


# In[15]:


#분류나 회귀를 위한 연습용 데이터의 구성 확인, 일반적으로 딕셔너리 형태

#load_iris()가 반환하는 객체의 키가 가리키는 값
print('\n feature_names의  type:',type(iris_data.feature_names))
print(' feature_names의 shape:',len(iris_data.feature_names))
print(iris_data.feature_names)

print('\n target_names의 type',type(iris_data.target_names))
print('target_names의 shape:',len(iris_data.target_names))
print(iris_data.target_names)

print('\n data의 type:',type(iris_data.target))
print('data의 shape:',iris_data.data.shape)
print(iris_data['data'])

print('\n target의 type:', type(iris_data.type))
print('target의 shape:',iris_data.target.shape)
print(iris_data.target)


# # Model Selection 모듈 소개
# 1. 학습/테스트 데이터 셋 분리 - train_test_split()  
#     - test_size:전체 데이터에서 테스트 데이터 세트 크기를 얼마로 샘플링? 디폴트는 25%
#     - train_size:전체 데이터에서 학습용 데이터를 얼마로 샘플링?
#     - shuffle: 데이터를 분리하기 전에 데이터를 섞을지? 디폴트는 True
#     - random_state: 호출할 때마다 동일한 학습/테스트용 데이터셋을 생성하기 위해 주어지는 값

# In[16]:


#예시
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# In[17]:


dt_clf = DecisionTreeClassifier()
iris_data = load_iris()

X_train,X_test ,y_train ,y_test = train_test_split(iris_data.data, iris_data.target,test_size=0.3,random_state=121)


# In[23]:


dt_clf.fit(X_train, y_train)
pred=dt_clf.predict(X_test)
print('예측 정확도:{0:.4f}'.format(accuracy_score(y_test,pred)))


# ### 교차검증
# > 데이터 편중을 막기 위해 여러 세트로 구성된 학습 데이터 세트와 검증 데이터 세트에서 학습과 평가를 수행
# >> 대부분의 모델 성능 평가는 교차검증 기반으로 1차 평가를 한 뒤에 최종적으로 테스트 데이터 셋에 적용해 평가함.
# 

# In[24]:


#K Fold 교차검증

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=156)

#5개의 폴드 세트로 분리하는 KFold 객체와 폴드 세트별 정확도를 담을 리스트 객체 생성
kfold=KFold(n_splits=5)
cv_accuracy=[]
#print('불꽃 데이터 세트 크기:',features.shape[0])


# In[25]:


#교차검증 수행 시마다 학습과 검증을 반복해 예측 정확도 측정 , split의 반환값도 구하기 

n_iter = 0
#KFold 객체의 split()를 호출하면 폴드 별 학습용  , 검증용 테스트의 로우 인덱스를 array로 반환
for train_index, test_index in kfold.split(features):
    #kfold.split()으로 반환된 인덱스를 이용해 학습용, 검증용 테스트 데이터 추출
    X_train , X_test =features[train_index],features[test_index]
    y_train, y_test = label[train_index],label[test_index]
    #학습 및 예측
    dt_clf.fit(X_train,y_train)
    pred = dt_clf.predict(X_test)
    n_iter+=1
    #반복 시마다 정확도 측정
    accuracy = np.round(accuracy_score(y_test,pred),4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('\n#{0} 교차 검증 정확도:{1},학습 데이터 크기:{2},검증 데이터 크기:{3}'.format(n_iter,accuracy , train_size,test_size))
    print('#{0} 검증 세트 인덱스:{1}'.format(n_iter,test_index))
    cv_accuracy.append(accuracy)
    
#개별 iteraction 별 정확도를 합하여 평균 정확도 계산
print('\n## 평균 검증 정확도:',np.mean(cv_accuracy))


# In[26]:


#Straited K Fold
#불균형한 분포도를 가진 레이블 데이터 집합을 위한 K 폴드

import pandas as pd


#레이블값의 분포도 확인
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data , columns = iris.feature_names)
iris_df['label']=iris.target
iris_df['label'].value_counts()


# In[27]:


#kfold로 실행
kfol= KFold(n_splits=3)
n_iter = 0
for train_index , test_index in kfold.split(iris_df):
    n_iter+=1
    label_train=iris_df['label'].iloc[train_index]
    label_test=iris_df['label'].iloc[test_index]
    print('## 교차검증: {0}'.format(n_iter))
    
    print('학습 레이블 데이터 분포:\n',label_train.value_counts())
    print('검증 레이블 데이터 분포:\n',label_test.value_counts())


# In[28]:


# StratifiedKFold 로 실행해보기
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=3)
n_iter=0

for train_index , test_index in skf.split(iris_df , iris_df['label']):
    n_iter += 1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print('##교차검증:{0}'.format(n_iter))
    print('학습 레이블 데이터 분포:\n',label_train.value_counts())
    print('검증 레이블 데이터 분포:\n',label_test.value_counts())


# #### 교차검증 도와주는 cross_val_score()
# - 폴드 세트를 설정
# - for 루프에서 반복으로 학습 및 테스트 데이터 인덱스 추출
# - 반복적으로 학습과 예측을 수행하고 예측 성능 반환
# > cross_val_scroe(estimator , X , y=None , scoring=None , cv=None , n_jobs=1 ,verbose=0, fit_params=None , pre_dispatch='2*n_jobs') 
# >> estimator , X , y , scoring , cv 가 주요 파라미터
# >>> X 는 피처데이터 , y는 레이블 데이터 scoring은 예측 성능 평가 지표 , cv는 교차 검증 폴드 수 

# In[29]:


from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.datasets import load_iris
import numpy as np

iris_data = load_iris()
dt_clf = DecisionTreeClassifier(random_state=156)

data= iris_data.data
label = iris_data.target


# In[30]:


#성능 지표는 정확도(accuracy), 교차 검증 세트는 3개

scores = cross_val_score(dt_clf , data, label , scoring='accuracy',cv=3)
print('교차 검증별 정확도:' , np.round(scores,4))
print('평균 검증 정확도:',np.round(np.mean(scores),4))


# ## GridSearchCV - 교차 검증과 최적 하이퍼 파라미터 튜닝을 한번에
# - 하이퍼 파라미터를 순차적으로 입력하면서 편리하게 최적의 파라미터를 도출

# In[1]:


grid_parameters = {'max_depth': [1,2,3],
                  'min_samples_split':[2,3]
                  }


# In[11]:


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#데이터를 로딩하고 학습 데이터와 테스트 데이터 분리

iris_data=load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target,test_size=0.2,random_state=121)

dtree = DecisionTreeClassifier()

### 파라미터를 딕셔너리 형태로 설정
parameters = {'max_depth':[1,2,3],'min_samples_split':[2,3]}


# In[7]:


#param_grid 의 하이퍼 파라미터를 3개의 train , test set fold 로 나누어 테스트 수행 설정.
### refit=True 가 default  
grid_dtree = GridSearchCV(dtree,param_grid=parameters, cv=3, refit=True)

#붓꽃 학습 데이터로 param_grid의 하이퍼 파라미터를 순차적으로 학습/평가
grid_dtree.fit(X_train , y_train)

#GridSearchCV 결과를 추출해 DataFrame으로 변환
scores_df=pd.DataFrame(grid_dtree.cv_results_)
scores_df[['params','mean_test_score','rank_test_score','split0_test_score','split1_test_score','split2_test_score']]


# - params 칼럼에는 수행할 때마다 적용된 개별 하이퍼 파라미터 값을 나타낸다
# - rank_test_score는 하이퍼 파라미터별로 성능이 좋은 score순위를 나타낸다 이때 1일때 최적의 하이퍼 파라미터
# - mean_test_score는 개별 하이퍼 파라미터별로 cv의 폴딩 테스트 세트에 대해 총 수행한 평가 평균값

# In[8]:


#최적 하이퍼 파라미터 값과 그때의 정확도
print('GridSearchCV 최적 파라미터: ',grid_dtree.best_params_)
print('GridsearchCV 최고 정확도:{0:.4f}'.format(grid_dtree.best_score_))


# In[12]:


#GridSearchCV의 refit으로 이미 학습된 estimator 반환
estimator = grid_dtree.best_estimator_

#GridSearchCV의 best_estimator_는 이미 최적 학습이 되어 별도 하급 필요 없음
pred=estimator.predict(X_test)
print('테스트 데이터 세트 정확도:{0:.4f}'.format(accuracy_score(y_test,pred)))


# In[ ]:




