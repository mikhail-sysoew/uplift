# About uplift
This Add-on to causalml library to measure period to period effect. Library compares two samples and measure uplift effect on target variable between them.

# Usage

## Base class
Class causal model is main class, which contains all methods, functions and temporary variables

### class causalmodel(model_type = None, feature_names=None, target_name=None, model_params=None)

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
      

### bias_elimination(caliper = None)

If control and target sample are biased, procedure make matching between them to eliminate this bias
Using Nearest Neigbors matching algorithm.

Parameters:

      -caliper (float) - measure of distance between examples in control and target variable to chain them in the pair. Input to nearest neighborrows algorithm
      
### fit_model ()

Fit model on variables, which was initialized in feature_names and target_name parameters in base class

### predict(x=None)

Predict method on new sample. Sample must contain only variables, which was initialized in feature_names in base class.
Returns uplift score.

Parameters:

      -x (dataframe) - dataframe, which should be scored 
      
### plot_tree ()

Vizualize decision tree. Model_type in base class should be equal 'tree' for use.

### shift_effect()

Calculate shift effect. This effect means share of uplift, that can be explained by variables shifting

## plot_variable_uplift (var=None, var_type=None, bins_num=None, ntile=None, raw_data=None)

Plot uplift prediction and variable distribution on target and control datasets

Parametes:

      - var (string) - name of the variable to plot from feature_names
      - var_type (string) - type (numeric or cat) should be set manually
      - raw_data (bool) - if True, plot will contain continuos x axis (only for numeric var_type)
      - bins_num (int) - number of bins for distribution
      - ntile (int) - number of equal parts of sample to plot. Only for raw_data = False

