import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

from playhouse.postgres_ext import ServerSide
from osp.corpus.models.text import Document_Text
import pickle

with open('model2.p', 'rb') as pf:
    full_clf = pickle.load(pf)

# Select all texts.
query = Document_Text.select()

# Mapping from a syllabus id to its predicted probability of being a syllabus
predictions = {}

# Counter
i = 0
skipped = 0
# Wrap the query in a server-side cursor, to avoid
# loading the plaintext for all docs into memory.
for sy in ServerSide(query):

    # Predict probability
    p  = full_clf.predict_proba([sy.text])[0,1]

    predictions[sy.document] = p
    if i < 5 or i % 10000 == 0:
        print('{}. {}: {}\tskipped {}'.format(i, sy.document, p, skipped))
    i += 1

with open('all_predictions.p', 'wb') as out:
    pickle.dump(predictions, out)
