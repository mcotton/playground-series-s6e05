import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


__all__ = [
    'copy_data', 
    'clean_data', 
    'remove_outliers', 
    'remove_duplicates', 
    'make_cluster_features', 
    'make_new_features', 
    'get_target', 
    'get_features', 
    'encode_all_the_things', 
    'fill_nas',
    'find_last_submission_file',
    'find_next_submission_file'
]

####
# Example Usage
###

# df = (train_df
#           .pipe(copy_data)
#           .pipe(clean_data)
#           .pipe(remove_outliers)
#           .pipe(remove_duplicates)
#           .pipe(make_new_features)
#           .pipe(encode_all_the_things)
#           .pipe(fill_nas)
#            )


def copy_data(df):
    return df.copy()


def clean_data(df):

    orig_feature_names = df.columns

    if 'id' in orig_feature_names:
        df = df.drop('id', axis=1)
    
    clean_feature_names = [i.replace(' ', '_').lower() for i in orig_feature_names]

    df = df.rename(columns=dict(zip(orig_feature_names, clean_feature_names)))
    
    if 'churn' in df.columns:
        df['churn'] = df['churn'].map({'Yes':1, 'No':0}).astype(int)
    
    return df

def remove_outliers(df):    
    return df


def remove_duplicates(df):
    df.drop_duplicates(inplace=True)
    return df


def get_target():
    target = 'pitnextlap'
    return target

def get_features(df):
    target = get_target()
    features = list(df.columns)
    
    if target in features:
        features.pop(features.index(target))
    
    return features

def make_cluster_features(df):

    combinations = [
        # (['Duration', 'Heart_Rate'], 10),
    ]

    for c in combinations:
    
        cluster_features = c[0]
        n_clusters = c[1]
        
        X = df[cluster_features]
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        
        df[f"{cluster_features[0]}_x_{cluster_features[1]}_cluster"] = labels

    return df

def make_new_features(df):

    # Example of different ways to one-hot encode
    # df = df.join(pd.get_dummies(df['Sex']))
    # df = df.drop('Sex', axis=1)
    # df['male'] = df['male'].apply(lambda x: 1 if x else 0)
    # df['female'] = df['female'].apply(lambda x: 1 if x else 0)

    # Example of creating feature from custom function
    # df['BMR'] = df.apply(calculate_BMR, axis=1)

    # Example of making cluster features
    df = make_cluster_features(df)
    
    target = get_target()
    features = get_features(df)

    # df['wet_race'] = df['compound'].map({'HARD': 0, 'MEDIUM': 0, 'SOFT': 0, 'INTERMEDIATE': 1, 'WET': 1})
    # df['avg_stint_per_race'] = df['race'].map(df.groupby('race')['stint'].median())

    df = df.drop('driver', axis=1)
    
    # # These featues hurt CV and LB, XGBoost didn't like them
    # df['study_x_attendance'] = df['study_hours'] * df['class_attendance']
    # df['study_hours_sq'] = df['study_hours'] ** 2
    # df['study_x_sleep'] = df['study_hours'] * df['sleep_hours']

    # # trying to identify the groups that miss by > +-30 points, slightly hurt score
    # df['very_low_effort'] = ((df['study_hours'] < 1) & (df['class_attendance'] < 60)).astype(int)
    # df['potential_ace'] = ((df['study_hours'] > 3) & (df['class_attendance'] > 70) & (df['sleep_hours'] > 7)).astype(int)

    # # High-error cases have ~44 lower study_hours * attendance, it slightly hurt score
    # df['study_attendance_multi'] = df['study_hours'] * df['class_attendance']


    # df['_study_hours_sin'] = np.sin(2 * np.pi * df['study_hours'] / 12).astype('float32')
    # df['_study_hours_cos'] = np.cos(2 * np.pi * df['study_hours'] / 12).astype('float32')
    # df['_class_attendance_sin'] = np.sin(2 * np.pi * df['class_attendance'] / 12).astype('float32')

    # df['study_hours_int'] = df['study_hours'].round().astype(int)
    # df['class_attendace_int'] = df['class_attendance'].round().astype(int)
    
    return df

def encode_all_the_things(df):
    target = get_target()
    
    return df

def fill_nas(df):
    
    # return df.fillna(df.median())

    
    
    return df


# def find_next_submission_file(data_dir='./archive'):
#     from glob import glob

#     for sub in sorted(glob(f"{data_dir}/submission_*.csv")):
#         print(sub)
#         i = sub.split('submission_')[1].split('.csv')[0]
#         next_file = f"{data_dir}/submission_{int(i)+1}.csv"
#         if int(i)+1 < 10:
#             next_file = f"{data_dir}/submission_0{int(i)+1}.csv"
        
#     return next_file

def find_last_submission_file(data_dir='./archive'):
    from glob import glob
    submissions = sorted(glob(f"{data_dir}/submission_*.csv"))

    sub = submissions[-1]
    
    return sub

def find_next_submission_file(data_dir='./archive'):
    sub = find_last_submission_file()
    
    i = sub.split('submission_')[1].split('.csv')[0]
    
    next_file = f"{data_dir}/submission_{int(i)+1}.csv"
    if int(i)+1 < 10:
        next_file = f"{data_dir}/submission_0{int(i)+1}.csv"
        
    return next_file