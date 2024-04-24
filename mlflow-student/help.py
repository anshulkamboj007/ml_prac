import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')

exp_id=mlflow.create_experiment('student_drop')

with mlflow.start_run(run_name='decisiontree') as run:
    mlflow.set_tag('version','1.0')
    pass

mlflow.end_run()

n_estimators=10
criterian='gini'

mlflow.log_param('n_estimators',n_estimators)
mlflow.log_param('criterian',criterian)
mlflow.log_metric('accuracy',0.9)