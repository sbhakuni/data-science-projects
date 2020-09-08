

import pandas as pd
import numpy as np
import nltk
import glob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from TenK import TenKDownloader, TenKScraper
import glob

# We will first do web scrapping for getting the text data of 10K form
with open('IT.txt','r') as f:
    company_name = [line.strip() for line in f]
    print (company_name)

downloader = TenKDownloader(company_name, '20090101','20191101')
downloader.download()


for folder in glob.glob('IT/*'):# we will have to do it sectorwise
    print(folder)
    for filename in glob.glob(folder+'/*.htm'):
        print ('FILE:', filename)
        scraper = TenKScraper('Item 1A', 'Item 1B') 
        scraper.scrape( filename, filename[:-3] + 'txt')

def tokenize(doc, lemmatized = False, no_stopword = False):
    tokens = nltk.word_tokenize(doc.lower()) 

    if no_stopword:

        stop = stopwords.words('english')
        tokens = [token for token in tokens if token not in stop]
    if lemmatized:

        tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens]

    return tokens

result = pd.DataFrame()
for folder in glob.glob('./IT/*'):

    for filename in glob.glob(folder+'/*.txt'):    
        with open(filename,'r',encoding="utf8") as f:
            doc = [line.strip() for line in f]

            df = pd.DataFrame([filename[7:-13],int(filename[-12:-4]),doc[0]]).T
            if len(doc[0]) > 10000:
                result = pd.concat([result, df])

Result = result.reset_index().rename(columns={0: "Ticker", 1: "date", 2: "text"})
Result.head()

#
rating = pd.read_excel('ITRating.xlsx')
Result['date'] = (Result['date']/100).astype(int)
Result['Ticker'] = Result.Ticker.str[6:]
rating['Data Date'] = np.around(rating['Data Date']/100).astype(int)


match = pd.merge(Result, rating, left_on = ['Ticker', 'date'], right_on = ['Ticker Symbol', 'Data Date'])
match.head()

dataset = match[['text', 'S&P Domestic Long Term Issuer Credit Rating']]
dataset.dropna(inplace = True)
Dataset = dataset.reset_index()


from sklearn.feature_extraction.text import TfidfVectorizer

# initialize the TfidfVectorizer 

tfidf_vect = TfidfVectorizer() 

tfidf_vect = TfidfVectorizer(stop_words="english") 

# generate tfidf matrix
dtm= tfidf_vect.fit_transform(Dataset["text"])

print("type of dtm:", type(dtm))
print("size of tfidf matrix:", dtm.shape)
dtm

#####tfid 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
# import method for split train/test data set
from sklearn.model_selection import train_test_split

# import method to calculate metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report

# split dataset into train (70%) and test sets (30%)
X_train, X_test, y_train, y_test = train_test_split(\
                dtm, Dataset['S&P Domestic Long Term Issuer Credit Rating'], test_size=0.4, random_state=0)

# train a multinomial naive Bayes model using the testing data
clf = MultinomialNB().fit(X_train, y_train)

# predict the new group for the test dataset
predicted=clf.predict(X_test)
print('Navie Bayes: ', accuracy_score(y_test, predicted))
print(classification_report(y_test, predicted))



#######SVM
from sklearn import svm

clf = svm.LinearSVC()
model_svm = clf.fit(X_train, y_train)
y_pred = model_svm.predict(X_test)
print('SVM: ', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#############RF
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
model_RF = clf.fit(X_train, y_train)
y_pred = model_RF.predict(X_test) 
print('RF: ', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#####MLP

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(100, 2), random_state=1)
model_RF = clf.fit(X_train, y_train)
y_pred = model_RF.predict(X_test) 
print('MLP: ', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))