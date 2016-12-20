# linguistic-invariant

### Goal
The goal of this project was to design a model that would capture features of an individual (a "fingerprint") and be used in combination to classify two individuals rapidly and accurately. Traditionally a training set generates one model that can only be used to classify a specific set of individuals, but with linguistic fingerprints each individual has a trained model with no information of the other individuals, so that at classification time any two fingerprints can be combined and run on the test data. I created various models and discuss their designs and strengths below.

### Basic Structure
fingerprinter.py is the runnable script specifying the names of the training and testing data sets. It interfaces with the FingerPrint class, which is a simple testing architecture for instantiating and running the model and calculating the test metrics. To choose a model, simply uncomment its import statement in FingerPrint.py:
```python
from VanillaModel import VanillaModel as Model
```
There are several points to the architecture. The models work like plugins, making it easy to switch between different models, as well as use and compare the code because the interface is exactly the same. They also do not share any code- each is completely self-contained. At the highest level, fingerprinter.py is not concerned with the models in any way; it simply calls `train()` and `test()`. Each model is also designed to be trained incrementally- that is, more training data can be added to it on the fly simply by calling `train()` again.

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

Running time will vary for all models, and accuracy and F-measure will vary for RandomWalk. There are several interesting things about this table. F-measure, which I used to balance recall and precision, happens to correlate well with accuracy, so henceforth I simply discuss the accuracy. The models are listed in the order in which I created them. 

Vanilla and Bigram are fairly fast because no POS tagging is performed. Pos and NaiveBayes use POS tagging for an accuracy boost and experience a serious slowdown, primarily in training. RandomWalk was a failed experiment, taking longer than Vanilla and performing significantly worse. Cue SimpleMatcher, which is significantly faster than Vanilla and Bigram in classification and also manages to be slightly more accurate than even Pos and NaiveBayes. And it's just 30 lines of code. PosVector attempts to make minor improvements to SimpleMatcher's accuracy.

### Explanation of Models
Vanilla was the baseline I submitted as the project proposal. As the name suggests, the fingerprint very plainly stores a frequency distribution of the words spoken and at classification time it scores based on "who said it more". Stop words and non-alpha tokens are filtered out for improved performance.

Bigram improves upon the model above, using a conditional frequency distribution to attempt to count the occurrences of the bigram, else use a backoff similar to Vanilla (unigram frequency). Stop words and non-alpha tokens are included for improved performance.

Pos uses NLTK's `nltk.pos_tag()` to tag the training and testing data. It stores the word and POS tag frequencies in two distributions, and continues the "who said it more" approach, using the word and tag as a pair. If the word-tag pair did not occur in the training set, the backoff model is to lookup just the tag, as this empirically performed better than looking up just the word.

NaiveBayes takes a more well-known approach by storing dual conditional frequency distributions, one for tag transitions (one tag to the next) and one for a tagged word output (tag to word). Instead of a cascade of backoffs, it calculates P(output) * P(transition).
```
argmax[person] P(output) = P(person | word,tag) = argmax[person] P(word,tag | person) / P(word,tag)
argmax[person] P(transition) = P(person | tag1,tag2) = argmax[person] P(tag1,tag2 | person) / P(tag1,tag2)
```
RandomWalk was a completely experimental approach, storing four conditional frequency distributions, for the previous two words and next two words of context. It uses the RandomWalker class to maintain a set of states that are predictions based on the current context (four words) and previous set of states (each with a numerical weight). These five sources of information are averaged into a frequency distribution. If the predictions (the most common words in the frequency distribution) contain the actual word, the score is incremented. The MAX_STATES parameter is set to 7 for improved performance. Further fine-tuning led me to remove large portions of code and yielded SimpleMatcher.

SimpleMatcher is very minimalistic- it considers only the previous word as the context and increments the score if the current word is in the frequency distribution given the context. The "transition" CFD is the same one used in Bigram.
```python
fd = self.word_transition_cfd[words[idx - 1]]
score += words[idx] in fd.keys()
```
For improved performance, tokens are not converted to lowercase, instead of `fd.most_common(MAX_STATES)` I use all words that appear as keys in the frequency distribution, and I don't increment by the count that appears in the distribution, just 1.

PosVector utilizes NumPy and vector space models to make a small improvement over SimpleMatcher. If the current word does not match any of the previous word's transitions, an added backoff model finds the cosine similarity of two part-of-speech vectors. One vector is the tag frequencies of the current word, the other vector is the average of the tag frequencies of the predicted words (all transitions from the previous word). At training time, the word-to-tag frequency map is created. NumPy provides the vector object.

### Conclusion
So SimpleMatcher and PosVector are the clear winners here. SimpleMatcher has a speed that is difficult to compare with. Bigram comes closest but in a real-time application, the classifying speed is more important as the models are trained in advance. Like Pos and NaiveBayes, PosVector benefits from annotated data (namely, part-of-speech tags) but it also is much faster than both models in classifying because it doesn't do POS tagging at classification time. So in applications where speed isn't the top concern, PosVector is the top contender, though only by a narrow margin. PosVector also has a lot of room for growth and improvement, itself an extension of SimpleMatcher which was limited in scope.

### Project Structure
All data is in the data/ directory. The three debate files have citations and each was grepped into another two files, one for Hillary and one for Trump. Every Python class has instructions on its use at the top, but everything is already setup to use the Vanilla Model, train on the first two debates, test on the third, and print the test metrics.