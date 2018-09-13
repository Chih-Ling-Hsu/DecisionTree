# Classification: Decision Tree

## Usage

### 1.	Import the class

```python
from DecisionTree import *
```


### 2.	Create an object

```python
dt = DecisionTree(method="C4.5", max_depth=3)
```

**Parameters -**
- **method** _(string)_	“ID3”,or “C4.5”
- **max_depth** _(int)_	the maximum depth of the tree


### 3.	Train a model

```python
dt.fit(X, y)
```
**Parameters -**
- **X** _(pandas.DataFrame)_	attributes of samples
- **y** _(pandas.DataFrame)_	labels of samples, must have the same number of rows as **X**
 

### 4.	Show the trained model

```python
dt.exportTree()
```

**Example output -**
```
--- [Outlook]
    --- overcast
        ===> yes
    --- rain
        --- [Wind]
            --- high
                ===> no
            --- low
                ===> yes
    --- sun
        --- [Humidity]
            --- <77.5
                ===> yes
            --- >=77.5
                ===> no
```

### 5.	Predict with the model

```python
y_pred = dt.predict(X)
```

**Parameters -**
- **X** _(pandas.DataFrame)_	attributes of samples

**Return -**
- **y_pred** _(list)_		prediction of labels for each sample


