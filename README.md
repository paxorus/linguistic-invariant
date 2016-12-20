# linguistic-invariant

### Goal
The goal of this project


### Basic Structure
fingerprinter.py is the runnable script specifying the names of the training and testing data sets. It interfaces with the FingerPrint class, which is a simple testing architecture for instantiating and running the model and calculating the test metrics. To choose a model, simply uncomment its import statement:
```python
from VanillaModel import VanillaModel as Model
```

### Benchmark Performance
Results from training on the first two debates and testing on the third are displayed in the table below.
| Model         | Training time (s) | Testing time (s) | Accuracy | F-measure |
|:-------------:|:-----------------:|:----------------:|:--------:|:---------:|
| Vanilla       | 0.817 | 1.587 | 78.261% | 70.339% |
| Bigram        | 0.823 | 1.1018 | 82.609% | 78.642% |
| Pos           | 52.982 | 30.286 | 83.540% | 78.543% |
| NaiveBayes    | 53.686 | 25.725 | 83.230% | 80.292% |
| RandomWalk    | 1.493 | 25.873 | 65.528% | 39.344% |
| SimpleMatcher | 1.063 | 0.333 | 83.851% | 79.688% |
| PosVector     | 54.200 | 7.151 | 84.161% | 80.755% |

Running time will vary for all models, and accuracy and F-measure will vary for RandomWalk. There are several interesting things about this table. The models are listed in the order in which I created them. Vanilla and Bigram are fairly fast because no POS tagging is performed. Pos and NaiveBayes use POS tagging for an accuracy boost and experience a serious slowdown, primarily in training. RandomWalk was a failed experiment, taking longer than Vanilla and performing significantly worse. Cue SimpleMatcher, which is significantly faster than Vanilla and Bigram in classification and also manages to be slightly more accurate than even Pos and NaiveBayes. And it's 30 lines of code. PosVector attempts to make minor improvements to SimpleMatcher's accuracy.

### Models
Vanilla w

### Findings
I used the F-measure to balance recall and precision.


### Project Structure
All data is in the data/ directory. The three debate files have citations and each was grepped into another two files, one for Hillary and one for Trump. Every Python class has instructions on its use at the top, but everything is already setup to use the Vanilla Model, train on the first two debates, test on the third, and print the test metrics.