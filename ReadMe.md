# Exploration of the feasibility and applicability of domain adaptation in machine learning-based code smell detection

## About
This repository provides an implementation of exploring the applicability of domain adaptation in machine learning-based code smell detection, as described in the paper:
**Exploration of the feasibility and applicability of domain adaptation in machine learning-based code smell detection**


Machine learning-based code smell detection was introduced to mitigate the limitations of the heuristic-based approach and the sub-jectivity issues. Due to limited choices of the publicly available datasets, most of the machine learning-based classifiers were trained by the earlier versions of open-source projects that no longer represent the characteristics and properties of modern programming languages. Our experiments exhibit the feasibility and applicability of using a machine learning classifier well-trained on the earlier versions of open-source projects to classify four types of code smells, i.e., god class, data class, feature envy, and long method, in modern Java open-source projects without extensive feature engineering. The performance produced by the supervised machine learning algorithms was evaluated and compared. Particle swarm optimization and Bayesian optimization were adopted to enhance the performance of the machine learning classifiers, i.e., decision tree and random forest. The experimental results indicated that it is feasible for the classifiers to identify the code smells though the performance is decreased in some cases. The hyper-parameter optimization slightly improves the performance of the machine learning classifiers when classifying feature envy, god class, and long method in a modern java project

### Paper 
Exploration of the feasibility and applicability of domain adaptation in machine learning-based code smell detection

### Machine Learning Algorithms 
* Decision tree (DT)
* Random forest (RF)

### Hyper-parameter Optimization Techniques 
* Particle Swarm Optimization (PSO)
* Bayesian Optimization with Tree-structure Parzen Estimator (BO-TPE)
* Bayesiam Optimization with Random forest (BO-RF/SMAC)

### Requirements
* Python 3.5+
* [scikit-learn](https://scikit-learn.org/stable/)
* [hyperopt](https://github.com/hyperopt/hyperopt)
* [optunity](https://github.com/claesenm/optunity)
* [SMAC3](https://github.com/automl/SMAC3)

## Citation
If you find this repository useful or being used within your research, please cite this paper as:

Sukkasem, P., Soomlek, C. (2023). Exploration of the Feasibility and Applicability of Domain Adaptation in Machine Learning-Based Code Smell Detection. In: Anutariya, C., Bonsangue, M.M. (eds) Data Science and Artificial Intelligence. DSAI 2023. Communications in Computer and Information Science, vol 1942. Springer, Singapore. https://doi.org/10.1007/978-981-99-7969-1_6

