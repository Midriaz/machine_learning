# Data load
# Document-term matrix
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from pymystem3 import Mystem
import re
import pickle
from nltk.corpus import stopwords


def lemmatize(word):
    # Lemmatization
    m = Mystem()
    return m.lemmatize(word)


def tokenize(row, pattern=r'[a-zA-Zа-яА-Я]+'):
    text = row['comment']
    words = re.findall(pattern, text)
    for i in range(len(words)):
        words[i] = lemmatize(words[i])

    if __name__ == '__main__':
        print('words done ', len(words))

    return ' '.join(words)


def do(df, stop_words):
    # we must remove stop words
    sw = stopwords.words('russian')
    with open(stop_words, encoding='utf-8') as f:
        our_stop_words = f.read().splitlines()
        f.close()

    sw = sw + our_stop_words

    # We will remove digits
    word_pattern = r'[a-zA-Zа-яА-Я]+'

    df['text'] = df.apply(tokenize, axis=1)

    # TOKENIZE
    cv = CountVectorizer(stop_words=sw, token_pattern=word_pattern)

    # Do a dataframe
    data_cv = cv.fit_transform(df['text'])
    data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())

    # save to pickle
    f = open('lemmatize.obj', 'wb')
    pickle.dump(data_dtm, f)
    f.close()

    return data_dtm


def term_matrix(df):
    # we must remove stop words
    sw = stopwords.words('russian')
    with open('comment_stop_words.txt', encoding='utf-8') as f:
        our_stop_words = f.read().splitlines()
        f.close()

    sw = sw + our_stop_words

    # We will remove digits
    word_pattern = r'[a-zA-Zа-яА-Я]+'

    cv = CountVectorizer(stop_words=sw, token_pattern=word_pattern)
    data_cv = cv.fit_transform(df)
    data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    del cv
    del data_cv

    f = open('lemmatize.obj', 'wb')
    pickle.dump(data_dtm, f)
    f.close()
    return data_dtm


if __name__ == '__main__':
    geo = pd.read_excel('geo_comment.xlsx')
    #geo.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
    #print(do(geo[['id', 'comment']], 'comment_stop_words.txt'))

    print(term_matrix(geo['comment']))
