# Disease Prediction using Machine Learning

## Project Overview
This project aims to predict the likelihood of various diseases based on a wide range of symptoms and health parameters. A machine learning model has been trained using several algorithms to determine the disease prognosis.

## Data Columns
The dataset includes the following input features (symptoms) and a target variable (prognosis):

- itching
- skin_rash
- nodal_skin_eruptions
- continuous_sneezing
- shivering
- chills
- joint_pain
- stomach_pain
- acidity
- ulcers_on_tongue
- muscle_wasting
- vomiting
- burning_micturition
- spotting_urination
- fatigue
- weight_gain
- anxiety
- cold_hands_and_feets
- mood_swings
- weight_loss
- restlessness
- lethargy
- patches_in_throat
- irregular_sugar_level
- cough
- high_fever
- sunken_eyes
- breathlessness
- sweating
- dehydration
- indigestion
- headache
- yellowish_skin
- dark_urine
- nausea
- loss_of_appetite
- pain_behind_the_eyes
- back_pain
- constipation
- abdominal_pain
- diarrhoea
- mild_fever
- yellow_urine
- yellowing_of_eyes
- acute_liver_failure
- fluid_overload
- swelling_of_stomach
- swelled_lymph_nodes
- malaise
- blurred_and_distorted_vision
- phlegm
- throat_irritation
- redness_of_eyes
- sinus_pressure
- runny_nose
- congestion
- chest_pain
- weakness_in_limbs
- fast_heart_rate
- pain_during_bowel_movements
- pain_in_anal_region
- bloody_stool
- irritation_in_anus
- neck_pain
- dizziness
- cramps
- bruising
- obesity
- swollen_legs
- swollen_blood_vessels
- puffy_face_and_eyes
- enlarged_thyroid
- brittle_nails
- swollen_extremeties
- excessive_hunger
- extra_marital_contacts
- drying_and_tingling_lips
- slurred_speech
- knee_pain
- hip_joint_pain
- muscle_weakness
- stiff_neck
- swelling_joints
- movement_stiffness
- spinning_movements
- loss_of_balance
- unsteadiness
- weakness_of_one_body_side
- loss_of_smell
- bladder_discomfort
- foul_smell_of_urine
- continuous_feel_of_urine
- passage_of_gases
- internal_itching
- toxic_look_(typhos)
- depression
- irritability
- muscle_pain
- altered_sensorium
- red_spots_over_body
- belly_pain
- abnormal_menstruation
- dischromic_patches
- watering_from_eyes
- increased_appetite
- polyuria
- family_history
- mucoid_sputum
- rusty_sputum
- lack_of_concentration
- visual_disturbances
- receiving_blood_transfusion
- receiving_unsterile_injections
- coma
- stomach_bleeding
- distention_of_abdomen
- history_of_alcohol_consumption
- fluid_overload.1
- blood_in_sputum
- prominent_veins_on_calf
- palpitations
- painful_walking
- pus_filled_pimples
- blackheads
- scurring
- skin_peeling
- silver_like_dusting
- small_dents_in_nails
- inflammatory_nails
- blister
- red_sore_around_nose
- yellow_crust_ooze
- **prognosis** (Target Variable)

## Machine Learning Models Used
The following machine learning models were implemented to predict disease prognosis:

```python
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "AdaboostClassifier":AdaBoostClassifier(),
    "Gradient Boosting Classifier":GradientBoostingClassifier(verbose=1),
    "Random Forest Classifier":RandomForestClassifier(verbose=1),
    "Support Vector Machine":SVC(verbose=True,probability=True),
    "K Nearest Neighbours":KNeighborsClassifier(),
    "Naive Bayes":GaussianNB(),
    "Catboost Classifier":CatBoostClassifier(verbose=1),
    "Logistic Regression": LogisticRegression(verbose=1),
    "XGBoost Classifier":XGBClassifier()
}
```


## Hyperparameter Tuning
RandomizedSearchCV was used to perform cross-validation and tune hyperparameters for some models:

```python
params = {
    'Logistic Regression':{
        'penalty':['elasticnet','l1','l2'],
        'C': [0.1, 1, 10],                     # Regularization strength
        'solver': ['saga', 'liblinear']        # Compatible solvers for elasticnet, l1, l2  
    },
    'Decision Tree':{
        'max_depth':[10,20,30],
        'min_samples_split':[2,5,10],
        'min_samples_leaf': [1, 2, 4]
    },
    'AdaboostClassifier':{
        'n_estimators':[50,100,150,200],
        'learning_rate':[0.1,0.01,0.001]
    },
    'Gradient Boosting Classifier':{
        'n_estimators':[100,150,200],
        'max_depth':[3,5,10],
        'learning_rate':[0.1,0.01,0.001]
    },
    'Random Forest Classifier':{
        'n_estimators': [100, 200, 450],       # Broader range of estimators
        'max_features': ['sqrt', 'log2'],      # Added 'sqrt' (default for classification)
        'max_depth': [50, 100, 340, None],     # Broader range and added None
        'min_samples_split': [2, 3, 5],        # Common starting values
        'min_samples_leaf': [1, 5, 10],        # Added smaller values for leaf nodes
        'criterion': ['gini', 'entropy']       # Added entropy as an option
    },
    'Support Vector Machine':{
        'kernel': ['linear', 'poly', 'sigmoid', 'rbf'],
        'gamma': ['scale', 'auto'],
        'C': [0.1, 1, 10]                      # Regularization parameter
    },
    'K Nearest Neighbours':{
        'n_neighbors': [3, 5, 7, 10],          # Added common k values
        'metric': ['euclidean', 'manhattan']   # Added manhattan for variety    
    },
    'Naive Bayes':{
        'var_smoothing': [1e-9, 1e-8, 1e-7]    # Common range for variance smoothing
    },
    'Catboost Classifier':{
        'iterations': [100, 200, 300],         # Number of boosting rounds
        'learning_rate': [0.1, 0.01, 0.001],
        'depth': [6, 8, 10],                   # Typical values for CatBoost depth
        'l2_leaf_reg': [1, 3, 5]               # Regularization parameter
    },
    "XGBoost Classifier":{
        'n_estimators': [100, 150, 200],       # Reasonable boosting rounds
        'learning_rate': [0.1, 0.01, 0.001],
        'max_depth': [3, 6, 10],               # Depth similar to Gradient Boosting
        'colsample_bytree': [0.6, 0.8, 1.0],  # Fraction of features for boosting
        'subsample': [0.8, 1.0]                # Fraction of samples for boosting
    }
}
```

## API Creation and Deployment
- The API was created using Flask to allow predictions based on user input.
- A Docker image has been created and is available on Docker Hub.

You can pull the Docker image using the following command:
```bash
docker pull gogetama/disease_prediction
```


## Future Enhancements
1. **Addition of More Models**: Further models such as LightGBM and AutoML frameworks can be explored for better performance.
2. **Feature Engineering**: Advanced techniques like feature interactions and polynomial features can be applied to improve the model’s accuracy.
3. **Mobile App**: Build a mobile-friendly interface for real-time disease prediction.
4. **Model Explainability**: Integrate tools like SHAP (SHapley Additive exPlanations) to interpret and visualize model predictions.

## References
1. [Scikit-learn](https://scikit-learn.org/)
2. [XGBoost Documentation](https://xgboost.readthedocs.io/)
3. [CatBoost Documentation]( https://catboost.ai/docs/)

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/Aadi1101/Disease_Prediction/blob/main/LICENSE) file for details.
