# About uplift
This Add-on to causalml library to measure period to period effect. Library compares two samples and measure uplift effect on target variable between them.

# Usage

##Base class
Class causal model is main class, which contains all methods, functions and temporary variables

 class causalmodel(model_type = None, feature_names=None, target_name=None, model_params=None)

Parameters:

      - model_type (string) - type of algorithm for uplift model {s-learner, t-learnet, random forest, tree}      
      - feature_names (list) - list with input features names       
      - target_name (string) - name of the target variable      
      - model_params (dictionary) - parametes for uplift base algorithm
      
      
      
## Methods

### preprocess()

Procedure, which makes 5 steps:

          1. Delete raws, where target is null
          2. Fill empty values with -1 for numeric and MISSING for categorial
          3. Fill outliers with >3 standart deviations with border value
          4. Make one-hot  on categorial features
          5. Write x,y and treatment into object from initial dataframe

Parameters:

      - None
      


      
