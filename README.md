# About uplift
This Add-on to causalml library to measure period to period effect. Library compares two samples and measure uplift effect on target variable between them.

# Usage
### Initialize model object

class causalmodel(model_type = None, feature_names=None, target_name=None, model_params=None)

Parameters:

      - model_type (string) - type of algorithm for uplift model {s-learner, t-learnet, random forest, tree}
      
      - feature_names (list) - list with input features names 
      
      - target_name (string) - name of the target variable
      
      - model_params (dictionary) - parametes for uplift base algorithm
      
      
