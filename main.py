import pandas
import sklearn.preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

bank = pandas.read_csv('bank.csv', sep=';')

boolean = { 'no': 0.0, 'yes': 1.0 }
months = {
    'jan': 1.0, 'feb': 2.0, 'mar': 3.0, 'apr': 4.0,  'may': 5.0,  'jun': 6.0,
    'jul': 7.0, 'aug': 8.0, 'sep': 9.0, 'oct': 10.0, 'nov': 11.0, 'dec': 12.0
}

bank.replace({
    'default': boolean,
    'housing': boolean,
    'loan':    boolean,
    'month':   months,
    'y':       boolean
}, inplace=True)

# Categorical features
#
# Since we plan to use logistic regression, add drop_first=True
# to use dummy instead of one-hot encoding

categorical = ['job', 'marital', 'education', 'contact', 'poutcome']
bank = pandas.get_dummies(bank, columns=categorical, prefix=categorical, drop_first=True)

# Numeric features
#
# Standardized because we plan to use KNN and SVM 

scaled = ['age', 'balance', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous']
bank[scaled] = sklearn.preprocessing.scale(bank[scaled].astype(float))

# Training set and targets
X = bank.drop(columns='y').values
t = bank['y'].values

#probmel 1 set aside 20% of the data as a test set
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2)

#problem 2 fit a naive bayes gaussianNB classifier
clf = GaussianNB()
clf.fit(X_train, t_train)

#problem 3
prediction = clf.predict(X_test)

#part a
score = clf.score(X_test,t_test)

# part b, confusion matrix
con_matrix = confusion_matrix(t_test,prediction)

#part c, plotting
pprob = clf.predict_proba(X_test)[:,1]
roc_score = roc_auc_score(t_test, pprob)
fpr, tpr, thresh = roc_curve(t_test,pprob)

plt.plot(fpr, tpr, label='roc_curve', color='red', lw=2)
plt.title("Roc")
plt.plot([0,1],[0,1], 'b')

#part d, compute roc_auc_score
print("Roc auc score: ", roc_score)

#problem 4

logReg = LogisticRegression(solver='lbfgs', fit_intercept=False)
X_train, X_train_logReg, t_train, t_train_logReg = train_test_split(X_train, t_train, test_size=0.2)
logReg.fit(X_train_logReg, t_train_logReg)
pprob_logReg = logReg.predict_proba(X_test)[:,1]
fpr_rt_lm, tpr_rt_lm, thresh_lm = roc_curve(t_test, pprob_logReg)

plt.plot(fpr_rt_lm, tpr_rt_lm, label='LogReg', color='purple', lw=2)
plt.plot([0,1],[0,1], 'b')


#problem 5
svc_lin = SVC(kernel='linear', probability=True, gamma='auto')
svc_score_lin = svc_lin.fit(X_train, t_train).decision_function(X_test)
fpr_lin, tpr_lin, thresh = roc_curve(t_test, svc_score_lin)

svc_pol = SVC(kernel='poly', probability=True, gamma='auto')
svc_score_pol = svc_pol.fit(X_train, t_train).decision_function(X_test)
fpr_pol, tpr_pol, thresholds = roc_curve(t_test, svc_score_pol)

svc_rbf = SVC(kernel='rbf', probability=True, gamma='auto')
svc_score_rbf = svc_rbf.fit(X_train, t_train).decision_function(X_test)
fpr_rbf, tpr_rbf, thresholds = roc_curve(t_test, svc_score_rbf)

svc_sigm = SVC(kernel='sigmoid', probability=True, gamma='auto')
svc_score_sigm = svc_sigm.fit(X_train, t_train).decision_function(X_test)
fpr_sig, tpr_sig, thresholds = roc_curve(t_test, svc_score_sigm)

#plot everything
plt.plot(fpr_lin, tpr_lin, label='Linear', color='green')
plt.plot(fpr_pol, tpr_pol, label='Poly', color='blue')
plt.plot(fpr_rbf, tpr_rbf, label='RBF', color='black')
plt.plot(fpr_sig, tpr_sig, label='Sigmoid', color='yellow')
plt.plot([0,1],[0,1], 'brown')
plt.legend(loc="lower right")

#problem 6
print("Logistics Regression has best score and performance")

#problem #7 ??

plt.show()
