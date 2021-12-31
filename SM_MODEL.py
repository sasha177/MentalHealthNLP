import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df_adm_notes_org_clean = pd.read_csv('/Users/sashamittal/Downloads/NLP project 10:22:19/df_adm_nlp_15_final.csv')

df_adm_notes_org_clean['OUTPUT_LABEL'] = (df_adm_notes_org_clean.DAYS_NEXT_ADMIT < 30).astype('int')
print('Number of positive samples:', (df_adm_notes_org_clean.OUTPUT_LABEL == 1).sum())
print('Number of negative samples:', (df_adm_notes_org_clean.OUTPUT_LABEL == 0).sum())
print('Total samples:', len(df_adm_notes_org_clean))

# shuffle the samples
df_adm_notes_org_clean = df_adm_notes_org_clean.sample(n=len(df_adm_notes_org_clean), random_state=42)
df_adm_notes_org_clean = df_adm_notes_org_clean.reset_index(drop=True)

# Save 30% of the data as validation and test data
df_valid_test = df_adm_notes_org_clean.sample(frac=0.30, random_state=42)
df_test = df_valid_test.sample(frac=0.5, random_state=42)
df_valid = df_valid_test.drop(df_test.index)

# use the rest of the data as training data
df_train_all = df_adm_notes_org_clean.drop(df_valid_test.index)

print('Test prevalence(n = %d):' % len(df_test), df_test.OUTPUT_LABEL.sum() / len(df_test))
print('Valid prevalence(n = %d):' % len(df_valid), df_valid.OUTPUT_LABEL.sum() / len(df_valid))
print('Train all prevalence(n = %d):' % len(df_train_all), df_train_all.OUTPUT_LABEL.sum() / len(df_train_all))
print('all samples (n = %d)' % len(df_adm_notes_org_clean))
assert len(df_adm_notes_org_clean) == (len(df_test) + len(df_valid) + len(df_train_all)), 'math didnt work'
# split the training data into positive and negative
rows_pos = df_train_all.OUTPUT_LABEL == 1
df_train_pos = df_train_all.loc[rows_pos]
df_train_neg = df_train_all.loc[~rows_pos]

# merge the balanced data
df_train = pd.concat([df_train_pos, df_train_neg.sample(n=len(df_train_pos), random_state=42)], axis=0)

# shuffle the order of training samples
df_train = df_train.sample(n=len(df_train), random_state=42).reset_index(drop=True)
print('Train prevalence (n = %d):' % len(df_train), df_train.OUTPUT_LABEL.sum() / len(df_train))


def preprocess_text(df):
    df.TEXT = df.TEXT.fillna(' ')
    df.TEXT = df.TEXT.str.replace('\n', ' ')
    df.TEXT = df.TEXT.str.replace('\r', ' ')
    return df


df_train = preprocess_text(df_train)
df_valid = preprocess_text(df_valid)
df_test = preprocess_text(df_test)
import nltk
from nltk import word_tokenize
#nltk.download('punkt')
import ssl

#try:
#    _create_unverified_https_context = ssl._create_unverified_context
#except AttributeError:
 #   pass
#else:
 #   ssl._create_default_https_context = _create_unverified_https_context

#nltk.download()


import string
print(string.punctuation)


def tokenizer_better(text):
    punc_list = string.punctuation + '0123456789'
    t = str.maketrans(dict.fromkeys(punc_list, " "))
    text = text.lower().translate(t)
    tokens = word_tokenize(text)
    return tokens
tokenizer_better('Test case with stars **')
sample_text = ['microbiology in science', 'predictive modeling in biology', 'NLP for president']

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(tokenizer = tokenizer_better)
vect.fit(sample_text)
X = vect.transform(sample_text)
X.toarray()
print(X)
vect.get_feature_names()

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(max_features = 350, tokenizer = tokenizer_better)
vect.fit(df_train.TEXT.values)
print(5)
neg_doc_matrix = vect.transform(df_train[df_train.OUTPUT_LABEL == 0].TEXT)
pos_doc_matrix = vect.transform(df_train[df_train.OUTPUT_LABEL == 1].TEXT)
neg_tf = np.sum(neg_doc_matrix,axis=0)
pos_tf = np.sum(pos_doc_matrix,axis=0)
neg = np.squeeze(np.asarray(neg_tf))
pos = np.squeeze(np.asarray(pos_tf))

term_freq_df = pd.DataFrame([neg,pos],columns=vect.get_feature_names()).transpose()
term_freq_df.columns = ['negative', 'positive']
term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['positive']
term_freq_df.sort_values(by='total', ascending=False).iloc[:10]

#Create a series from the sparse matrix
d = pd.Series(term_freq_df.total,
              index = term_freq_df.index).sort_values(ascending=False)
ax = d[:50].plot(kind='bar', figsize=(10,6), width=.8, fontsize=14, rot=90,color = 'b')
ax.title.set_size(18)
plt.ylabel('count')
plt.show()
ax = d[50:100].plot(kind='bar', figsize=(10,6), width=.8, fontsize=14, rot=90,color = 'b')
ax.title.set_size(18)
plt.ylabel('count')
plt.show()
my_stop_words = ['the','and','to','of','was','with','a','c','r','on','in','for','name',
                 'is','patient','s','he','at','as','or','one','she','his','her','am',
                 'were','you','pt','pm','by','be','had','your','this','date',
                'from','there','an','that','p','are','have','has','h','but','o',
                'namepattern','which','every','also']
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(max_features = 350,
                       tokenizer = tokenizer_better,
                       stop_words = my_stop_words)
# this could take a while
vect.fit(df_train.TEXT.values)
print(4)

X_train_tf = vect.transform(df_train.TEXT.values)
print(4)

X_valid_tf = vect.transform(df_valid.TEXT.values)
print(4)

y_train = df_train.OUTPUT_LABEL
print(4)

y_valid = df_valid.OUTPUT_LABEL
print(4)


from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(C = 0.001, penalty = 'l2', random_state = 42)
clf.fit(X_train_tf, y_train)
model = clf
y_train_preds = model.predict_proba(X_train_tf)[:,1]
y_valid_preds = model.predict_proba(X_valid_tf)[:,1]
print(y_train[:10].values)
print(y_train_preds[:10])
def calc_accuracy(y_actual, y_pred, thresh):
    # this function calculates the accuracy with probability threshold at thresh
    return (sum((y_pred > thresh) & (y_actual == 1))+sum((y_pred < thresh) & (y_actual == 0))) /len(y_actual)

def calc_recall(y_actual, y_pred, thresh):
    # calculates the recall
    return sum((y_pred > thresh) & (y_actual == 1)) /sum(y_actual)

def calc_precision(y_actual, y_pred, thresh):
    # calculates the precision
    return sum((y_pred > thresh) & (y_actual == 1)) /sum(y_pred > thresh)

def calc_specificity(y_actual, y_pred, thresh):
    # calculates specificity
    return sum((y_pred < thresh) & (y_actual == 0)) /sum(y_actual ==0)

def calc_prevalence(y_actual):
    # calculates prevalence
    return sum((y_actual == 1)) /len(y_actual)
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_preds)
fpr_valid, tpr_valid, thresholds_valid = roc_curve(y_valid, y_valid_preds)


thresh = 0.5
print(5)
auc_train = roc_auc_score(y_train, y_train_preds)
auc_valid = roc_auc_score(y_valid, y_valid_preds)

print('Train AUC:%.3f'%auc_train)
print('Valid AUC:%.3f'%auc_valid)

print('Train accuracy:%.3f'%calc_accuracy(y_train, y_train_preds, thresh))
print('Valid accuracy:%.3f'%calc_accuracy(y_valid, y_valid_preds, thresh))


print('Train recall:%.3f'%calc_recall(y_train, y_train_preds, thresh))
print('Valid recall:%.3f'%calc_recall(y_valid, y_valid_preds, thresh))

print('Train precision:%.3f'%calc_precision(y_train, y_train_preds, thresh))
print('Valid precision:%.3f'%calc_precision(y_valid, y_valid_preds, thresh))

print('Train specificity:%.3f'%calc_specificity(y_train, y_train_preds, thresh))
print('Valid specificity:%.3f'%calc_specificity(y_valid, y_valid_preds, thresh))

print('Train prevalence:%.3f'%calc_prevalence(y_train))
print('Valid prevalence:%.3f'%calc_prevalence(y_valid))


plt.plot(fpr_train, tpr_train,'r-', label = 'Train AUC: %.2f'%auc_train)
plt.plot(fpr_valid, tpr_valid,'b-',label = 'Valid AUC: %.2f'%auc_valid)
plt.plot([0,1],[0,1],'-k')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


def get_most_important_features(vectorizer, model, n=5):
    index_to_word = {v: k for k, v in vectorizer.vocabulary_.items()}

    # loop for each class
    classes = {}
    for class_index in range(model.coef_.shape[0]):
        word_importances = [(el, index_to_word[i]) for i, el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key=lambda x: x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key=lambda x: x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops': tops,
            'bottom': bottom
        }
    return classes


def plot_important_words(top_scores, top_words, bottom_scores, bottom_words, name):
    y_pos = np.arange(len(top_words))
    top_pairs = [(a, b) for a, b in zip(top_words, top_scores)]
    top_pairs = sorted(top_pairs, key=lambda x: x[1])

    bottom_pairs = [(a, b) for a, b in zip(bottom_words, bottom_scores)]
    bottom_pairs = sorted(bottom_pairs, key=lambda x: x[1], reverse=True)

    top_words = [a[0] for a in top_pairs]
    top_scores = [a[1] for a in top_pairs]

    bottom_words = [a[0] for a in bottom_pairs]
    bottom_scores = [a[1] for a in bottom_pairs]
    fig = plt.figure(figsize=(10, 15))

    plt.subplot(121)
    plt.barh(y_pos, bottom_scores, align='center', alpha=0.5)
    plt.title('Negative', fontsize=20)
    plt.yticks(y_pos, bottom_words, fontsize=14)
    plt.suptitle('Key words', fontsize=16)
    plt.xlabel('Importance', fontsize=20)

    plt.subplot(122)
    plt.barh(y_pos, top_scores, align='center', alpha=0.5)
    plt.title('Positive', fontsize=20)
    plt.yticks(y_pos, top_words, fontsize=14)
    plt.suptitle(name, fontsize=16)
    plt.xlabel('Importance', fontsize=20)

    plt.subplots_adjust(wspace=0.8)
    plt.show()
    importance = get_most_important_features(vect, clf, 50)

    top_scores = [a[0] for a in importance[0]['tops']]
    top_words = [a[1] for a in importance[0]['tops']]
    bottom_scores = [a[0] for a in importance[0]['bottom']]
    bottom_words = [a[1] for a in importance[0]['bottom']]
    plot_important_words(top_scores, top_words, bottom_scores, bottom_words, "Most important words")

    my_new_stop_words = ['the', 'and', 'to', 'of', 'was', 'with', 'a', 'on', 'in', 'for', 'name',
                         'is', 'patient', 's', 'he', 'at', 'as', 'or', 'one', 'she', 'his', 'her', 'am',
                         'were', 'you', 'pt', 'pm', 'by', 'be', 'had', 'your', 'this', 'date',
                         'from', 'there', 'an', 'that', 'p', 'are', 'have', 'has', 'h', 'but', 'o','q',
                         'namepattern', 'which', 'every', 'also', 'should', 'if', 'it', 'been', 'who', 'during', 'x', 'admit', 'dat', 'discharg']
    vect = CountVectorizer(lowercase=True, max_features=350, tokenizer=tokenizer_better, stop_words=my_new_stop_words)
    vect.fit(df_train.TEXT.values)

    X_train_tf = vect.transform(df_train.TEXT.values)
    X_valid_tf = vect.transform(df_valid.TEXT.values)
    y_train = df_train.OUTPUT_LABEL
    y_valid = df_valid.OUTPUT_LABEL
    import numpy as np
    from sklearn.model_selection import learning_curve
    from sklearn.model_selection import ShuffleSplit

    def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("AUC")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='roc_auc')
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="b")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="b",
                 label="Cross-validation score")

        plt.legend(loc="best")
        return plt

    title = "Learning Curves (Logistic Regression)"
    # Cross validation with 5 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    estimator = LogisticRegression(C=0.001, penalty='l2')
    plot_learning_curve(estimator, title, X_train_tf, y_train, ylim=(0.2, 1.01), cv=cv, n_jobs=4)

    plt.show()
    from sklearn.linear_model import LogisticRegression

    Cs = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003]
    train_aucs = np.zeros(len(Cs))
    valid_aucs = np.zeros(len(Cs))

    for ii in range(len(Cs)):
        C = Cs[ii]
        print('\n C:', C)

        # logistic regression

        clf = LogisticRegression(C=C, penalty='l2', random_state=42)
        clf.fit(X_train_tf, y_train)

        model = clf
        y_train_preds = model.predict_proba(X_train_tf)[:, 1]
        y_valid_preds = model.predict_proba(X_valid_tf)[:, 1]

        auc_train = roc_auc_score(y_train, y_train_preds)
        auc_valid = roc_auc_score(y_valid, y_valid_preds)
        print('Train AUC:%.3f' % auc_train)
        print('Valid AUC:%.3f' % auc_valid)
        train_aucs[ii] = auc_train
        valid_aucs[ii] = auc_valid
        plt.plot(Cs, train_aucs, 'bo-', label='Train')
        plt.plot(Cs, valid_aucs, 'ro-', label='Valid')
        plt.legend()
        plt.xlabel('Logistic Regression - C')
        plt.ylabel('AUC')
        plt.show()
        num_features = [100, 300, 1000, 3000, 10000, 30000]
        train_aucs = np.zeros(len(num_features))
        valid_aucs = np.zeros(len(num_features))

        for ii in range(len(num_features)):
            num = num_features[ii]
            print('\nnumber of features:', num)
            vect = CountVectorizer(lowercase=True, max_features=num,
                                   tokenizer=tokenizer_better, stop_words=my_new_stop_words)

            # This could take a while
            vect.fit(df_train.TEXT.values)

            X_train_tf = vect.transform(df_train.TEXT.values)
            X_valid_tf = vect.transform(df_valid.TEXT.values)
            y_train = df_train.OUTPUT_LABEL
            y_valid = df_valid.OUTPUT_LABEL

            clf = LogisticRegression(C=0.001, penalty='l2', random_state=42)
            clf.fit(X_train_tf, y_train)

            model = clf
            y_train_preds = model.predict_proba(X_train_tf)[:, 1]
            y_valid_preds = model.predict_proba(X_valid_tf)[:, 1]

            auc_train = roc_auc_score(y_train, y_train_preds)
            auc_valid = roc_auc_score(y_valid, y_valid_preds)
            print('Train AUC: %.3f' % auc_train)
            print('Valid AUC:%.3f' % auc_valid)
            train_aucs[ii] = auc_train
            valid_aucs[ii] = auc_valid
            plt.plot(num_features, train_aucs, 'bo-', label='Train')
            plt.plot(num_features, valid_aucs, 'ro-', label='Valid')
            plt.legend()
            plt.xlabel('Number of features')
            plt.ylabel('AUC')
            plt.show()
            num_features = [100, 300, 1000, 3000, 10000, 30000]
            train_aucs = np.zeros(len(num_features))
            valid_aucs = np.zeros(len(num_features))

            for ii in range(len(num_features)):
                num = num_features[ii]
                print('\nnumber of features:', num)
                vect = CountVectorizer(lowercase=True, max_features=num,
                                       tokenizer=tokenizer_better, stop_words=my_new_stop_words)

                # This could take a while
                vect.fit(df_train.TEXT.values)

                X_train_tf = vect.transform(df_train.TEXT.values)
                X_valid_tf = vect.transform(df_valid.TEXT.values)
                y_train = df_train.OUTPUT_LABEL
                y_valid = df_valid.OUTPUT_LABEL

                clf = LogisticRegression(C=0.001, penalty='l2', random_state=42)
                clf.fit(X_train_tf, y_train)

                model = clf
                y_train_preds = model.predict_proba(X_train_tf)[:, 1]
                y_valid_preds = model.predict_proba(X_valid_tf)[:, 1]

                auc_train = roc_auc_score(y_train, y_train_preds)
                auc_valid = roc_auc_score(y_valid, y_valid_preds)
                print('Train AUC: %.3f' % auc_train)
                print('Valid AUC:%.3f' % auc_valid)
                train_aucs[ii] = auc_train
                valid_aucs[ii] = auc_valid

            plt.plot(num_features, train_aucs, 'bo-', label='Train')
            plt.plot(num_features, valid_aucs, 'ro-', label='Valid')
            plt.legend()
            plt.xlabel('Number of features')
            plt.ylabel('AUC')
            plt.show()
            # shuffle the samples

            rows_not_death = df_adm_notes_org_clean.DEATHTIME.isnull()

            df_adm_notes_not_death = df_adm_notes_org_clean.loc[rows_not_death].copy()
            df_adm_notes_not_death = df_adm_notes_not_death.sample(n=len(df_adm_notes_not_death), random_state=42)
            df_adm_notes_not_death = df_adm_notes_not_death.reset_index(drop=True)

            # Save 30% of the data as validation and test data
            df_valid_test = df_adm_notes_not_death.sample(frac=0.30, random_state=42)

            df_test = df_valid_test.sample(frac=0.5, random_state=42)
            df_valid = df_valid_test.drop(df_test.index)

            # use the rest of the data as training data
            df_train_all = df_adm_notes_not_death.drop(df_valid_test.index)

            print('Test prevalence(n = %d):' % len(df_test), df_test.OUTPUT_LABEL.sum() / len(df_test))
            print('Valid prevalence(n = %d):' % len(df_valid), df_valid.OUTPUT_LABEL.sum() / len(df_valid))
            print('Train all prevalence(n = %d):' % len(df_train_all),
                  df_train_all.OUTPUT_LABEL.sum() / len(df_train_all))
            print('all samples (n = %d)' % len(df_adm_notes_org_clean))
            assert len(df_adm_notes_not_death) == (len(df_test) + len(df_valid) + len(df_train_all)), 'math didnt work'

            # split the training data into positive and negative
            rows_pos = df_train_all.OUTPUT_LABEL == 1
            df_train_pos = df_train_all.loc[rows_pos]
            df_train_neg = df_train_all.loc[~rows_pos]

            # merge the balanced data
            df_train = pd.concat([df_train_pos, df_train_neg.sample(n=len(df_train_pos), random_state=42)], axis=0)

            # shuffle the order of training samples
            df_train = df_train.sample(n=len(df_train), random_state=42).reset_index(drop=True)

            print('Train prevalence (n = %d):' % len(df_train), df_train.OUTPUT_LABEL.sum() / len(df_train))

            # preprocess the text to deal with known issues
            df_train = preprocess_text(df_train)
            df_valid = preprocess_text(df_valid)
            df_test = preprocess_text(df_test)
            my_new_stop_words = ['the', 'and', 'to', 'of', 'was', 'with', 'a', 'on','q','l', 'in', 'for', 'name',
                                 'is', 'patient', 's', 'he', 'at', 'as', 'or', 'one', 'she', 'his', 'her', 'am',
                                 'were', 'you', 'pt', 'pm', 'by', 'be', 'had', 'your', 'this', 'date',
                                 'from', 'there', 'an', 'that', 'p', 'are', 'have', 'has', 'h', 'but', 'o',
                                 'namepattern', 'which', 'every', 'also', 'should', 'if', 'it', 'been', 'who', 'during',
                                 'x', 'when']

            from sklearn.feature_extraction.text import CountVectorizer

            vect = CountVectorizer(lowercase=True, max_features=350,
                                   tokenizer=tokenizer_better,
                                   stop_words=my_new_stop_words)

            # This could take a while
            vect.fit(df_train.TEXT.values)

            X_train_tf = vect.transform(df_train.TEXT.values)
            X_valid_tf = vect.transform(df_valid.TEXT.values)
            X_test_tf = vect.transform(df_test.TEXT.values)

            y_train = df_train.OUTPUT_LABEL
            y_valid = df_valid.OUTPUT_LABEL
            y_test = df_test.OUTPUT_LABEL

            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_curve
            from sklearn.metrics import roc_auc_score

            clf = LogisticRegression(C=0.001, penalty='l2', random_state=42)
            clf.fit(X_train_tf, y_train)

            model = clf
            y_train_preds = model.predict_proba(X_train_tf)[:, 1]
            y_valid_preds = model.predict_proba(X_valid_tf)[:, 1]
            y_test_preds = model.predict_proba(X_test_tf)[:, 1]
            fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_preds)
            fpr_valid, tpr_valid, thresholds_valid = roc_curve(y_valid, y_valid_preds)
            fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_preds)

            thresh = 0.5

            auc_train = roc_auc_score(y_train, y_train_preds)
            auc_valid = roc_auc_score(y_valid, y_valid_preds)
            auc_test = roc_auc_score(y_test, y_test_preds)

            print('Train prevalence(n = %d): %.3f' % (len(y_train), sum(y_train) / len(y_train)))
            print('Valid prevalence(n = %d): %.3f' % (len(y_valid), sum(y_valid) / len(y_valid)))
            print('Test prevalence(n = %d): %.3f' % (len(y_test), sum(y_test) / len(y_test)))

            print('Train AUC:%.3f' % auc_train)
            print('Valid AUC:%.3f' % auc_valid)
            print('Test AUC:%.3f' % auc_test)

            print('Train accuracy:%.3f' % calc_accuracy(y_train, y_train_preds, thresh))
            print('Valid accuracy:%.3f' % calc_accuracy(y_valid, y_valid_preds, thresh))
            print('Test accuracy:%.3f' % calc_accuracy(y_test, y_test_preds, thresh))

            print('Train recall:%.3f' % calc_recall(y_train, y_train_preds, thresh))
            print('Valid recall:%.3f' % calc_recall(y_valid, y_valid_preds, thresh))
            print('Test recall:%.3f' % calc_recall(y_test, y_test_preds, thresh))

            print('Train precision:%.3f' % calc_precision(y_train, y_train_preds, thresh))
            print('Valid precision:%.3f' % calc_precision(y_valid, y_valid_preds, thresh))
            print('Test precision:%.3f' % calc_precision(y_test, y_test_preds, thresh))

            print('Train specificity:%.3f' % calc_specificity(y_train, y_train_preds, thresh))
            print('Valid specificity:%.3f' % calc_specificity(y_valid, y_valid_preds, thresh))
            print('Test specificity:%.3f' % calc_specificity(y_test, y_test_preds, thresh))

            plt.plot(fpr_train, tpr_train, 'r-', label='Train AUC: %.2f' % auc_train)
            plt.plot(fpr_valid, tpr_valid, 'b-', label='Valid AUC: %.2f' % auc_valid)
            plt.plot(fpr_test, tpr_test, 'g-', label='Test AUC: %.2f' % auc_test)

            plt.plot([0, 1], [0, 1], '-k')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.show()