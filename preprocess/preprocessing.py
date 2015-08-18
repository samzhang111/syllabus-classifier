# coding: utf-8
import pandas as pd
df = pd.read_csv('/home/ubuntu/data/raw/syllabus-refinement_csv.csv')
df['syllabus'] = df.tags.str.match("Yes, it's a Syllabus")
df2 = df[['title', 'text', 'syllabus']]
df2.to_csv('refinement.csv', index=False)

raw_training_df = pd.read_csv('/home/ubuntu/data/raw/osp-tagging.csv')
training_df = raw_training_df[raw_training_df.tags.apply(
    lambda x: isinstance(x, str)) & ~raw_training_df.text.isnull()]
training_df.reset_index(inplace=True)

def is_syllabus_tag(tag):
    try:
        return 'syllabus' in tag.lower() and 'not syllabus' not in tag.lower()
    except AttributeError:
        return False

is_syllabus = training_df.tags.apply(is_syllabus_tag)
training_df['syllabus'] = is_syllabus
training_df[['title', 'text', 'syllabus']].to_csv('syllabus_tags.csv', index=False)
