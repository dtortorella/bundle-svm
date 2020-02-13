# Bundle SVM

Support vector machine with bundle method and span vector selection.

We provide an *example.m* script to run the optimization algorithm code in a simple enviroment of a regression task with MLCUP17 dataset; *MLexample.m* script to run a model selection on different datasets for classification and regression models.

Every function is well documented in their own matlab file.

## Directory structure

* *src/*: contains the developed code and scripts

* *data/*: contains dataset used for the experiments

* *experiments/*: contains experiment scripts

* *example.m* : script with example calls to main projects code. It runs an SVR task solving it with the bundle methods and quadprog, givin possibility of comparisons. It also runs some span selection algorithms to appreciate the mesurements on obtained reduced Gram matrix

* *ml_example.m*: script to run model selection for SVM and SVR datasets

### Optimization algorithms
* *bundleizator.m*: solves a generic SVM/SVR selecting a vector basis of the Gram matrix and then applying a bundle method to the problem

* *bundleizator_pruning.m*: same algorithm as *bundleizator.m* for solving the problem. It reduces the number of subgradients deleting inactive one

* *big_fat_solver.m*: solves the QP  problem with a single quadprog call

### Span selection algorithms
* *select_span_vectors.m*: selects the span vectors of a Gram matrix with diffent algorithms

* *sRRQR_tol.m*, *sRRQR_rank.m*, *sRRQR.m*: strong Rank Revealing QR factorization algorithm implemented by Xin Xing

* *iqr.m*: gives a rank-m QR factorization of a matrix approximating m with repeated incremental matlab qr factorizations

* *isvd.m*: rank-m SVD factorization of a matrix approximating m with repeated incremental matlab svd factorizations

* *eval_parallelity.m*: metric to evaluate a basis parallelity to a vector space

* *eval_orthonormality.m*: metric to evaluate the orthonormality of a basis

### Experiments on bundleizator and span selection

* *exper_clock.m*:  execution time comparisons
* *exper_span_full.m*: span selection with full factorization evaluation
* *exper_span_incr.m*: span selection with incremental factorization evaluation
* *exper_pruning.m*: time and solution vector evaluation

### Machine learning

*svm_select_model.m*, *svm_train.m*, *svm_predict.m*, *svm_select_model_bayesianely.m*: implement model selection, training and prediction for a classification task.

*svr_select_model.m*,*svr_train.m*, *svr_predict.m*, *svr_select_model_bayesianely.m*: implement model selection, training and prediction for a regression task.

### Utility functions
* *einsensitive_loss.m , einsensitive_dloss.m*, *hinge_loss.m , hinge_dloss.m*: implementation of losses and their derivatives

* *gram_matrix.m*, *gram_matrix2.m*, *gram_norm_matrix.m*, *gram_norm_matrix2.m*: utility to generate a Gram matrix given dataset and a kernel function.

