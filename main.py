# Compare challenger implementation of feature streaming RF to the baseline.
# Beware, some data sets are in OpenML multiple times (they have the same name but different did,...).
# This may affect, for example, hypothesis testing (we will get overconfident).

import openml   # Requires scikit <= 0.21
import pandas as pd
import psycopg2
from numpy import *
from sklearn.model_selection import train_test_split


# Setting
from rf_baseline import RFBaseline
from rf_challenger import RFChallenger
from rf_offline import offline

mtry = 2./3.    # We use rejection sampling -> for the sake of the speed of sampling, we use high mtry
n_jobs = 30     # Count of trees in a generation
np.set_printoptions(precision=2)


# Database setting
connection = psycopg2.connect(user="jan",
                              password="",
                              host="127.0.0.1",
                              port="5432",
                              database="PredictorFactory")
cursor = connection.cursor()
postgres_insert_query = """ INSERT INTO predictor_factory.rf_njob30 (dataset, did, seed, auc_challenger, auc_baseline, auc_offline) VALUES (%s,%s,%s,%s,%s,%s)"""


# Dataset list
openml_list = openml.datasets.list_datasets()
datalist = pd.DataFrame.from_dict(openml_list, orient='index')
datalist = datalist[['did', 'name', 'NumberOfInstances', 'NumberOfFeatures', 'NumberOfClasses']]
filtered = datalist.query('NumberOfClasses == 2')
filtered = filtered.query('NumberOfInstances < 200000')
filtered = filtered.query('NumberOfInstances > 20')
filtered = filtered.query('NumberOfFeatures < 15')
filtered = filtered.query('did >= 1')
filtered = filtered.query('did != 376') # Because of sparse encoding
filtered = filtered.query('did < 40945') # Because PyOpenML cannot handle string features


for did in filtered.did:

        # Download dataset
        dataset = openml.datasets.get_dataset(did)
        X_in, y_in = dataset.get_data(target=dataset.default_target_attribute)
        X_in[np.isnan(X_in)] = -1  # Very simple missing value treatment
        print('Dataset', dataset.name, did, flush=True) # For progress indication


        for seed in range(2101, 2111):
            # Random feature permutation, which defines the order of features in the stream
            features = np.random.RandomState(seed=seed).permutation(np.size(X_in, 1))
            print(features)
            X_perm = X_in[:, features]

            # Training/testing split
            X, X_t, y, y_t = train_test_split(X_perm, y_in, test_size=1./3., random_state=seed, stratify=y_in)

            # Evaluate the challenger&baseline model at the last step
            challenger = RFChallenger(y=y, X_t=X_t, y_t=y_t, n_jobs=n_jobs, mtry=mtry, random_state=seed)
            baseline = RFBaseline(y=y, X_t=X_t, y_t=y_t, n_jobs=n_jobs, mtry=mtry, random_state=seed)
            for generation in range(np.size(X, 1)):
                challenger.addFeature(X[:, generation])
                baseline.addFeature(X[:, generation])

            auc_challenger = challenger.getAUC()
            auc_baseline = baseline.getAUC()

            # Evaluate the offline model (all features without any previous tree)
            auc_offline = offline(X, y, X_t, y_t, n_jobs, mtry, seed)

            # Write to db
            record_to_insert = (dataset.name, dataset.dataset_id, seed, auc_challenger, auc_baseline, auc_offline)
            cursor.execute(postgres_insert_query, record_to_insert)
            connection.commit()


# Clean up
cursor.close()
connection.close()
print("PostgreSQL connection is closed")




