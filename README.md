# Part of Speech Tagging
POS tagging is a process where each word in a sentence is given a tag based on its definition aswell as context. The dataset used to train the Hidden Markov Model was the Wall Street Journal dataset of the Penn Treebank.

## Vocabulary Creation
One of the most important tasks when training HMM is the creation of a decent vocabulary that can handle unknown words. As such, I utilized a simple strategy where words whose occurences is less than a threshold (3 in my experiments) were replaced with the <code>/<UNK/></code> tag. Consequently, this allowed the model build a decent intuition on how to handle such cases.

## Training
The crux of the learning is calculating the transition and emission parameters in HMM. Luckily, it is as simple as counting occurences and pairs. Furthermore, I applied LaPlace smoothing so that the probabilities are not 0, which prevents the cases where it might underflow. After training the model, two decoding algorithms were compared: Greedy and Viterbi. Greedy is simple; the model always picks the locally best which is not necessarily the optimal best overall. This is a fast algorithm but may yield suboptimal solutions. On the other hand, Viterbi decoding uses dynamic programming to attempt to find the optimal solution, which is more expensive computationally. Having said that, Viterbi should yield better results.

## Results
| Model | Accuracy |
| ----- | -------- |
| HMM + Greedy Decoding | 94.4% |
| HMM + Viterbi Decoding | 95.5% |
