# Course projects: Vertical Federated Learning

## Multi-Participant Multi-Class Vertical Federated Learning
### link: https://arxiv.org/pdf/2001.11154.pdf

### Description:
Typically, vertical federated learning (VFL) involves one party possessing the labels with the features being partitioned across multiple participants. This paper provides a privacy-preserving label-sharing algorithm to enable joint learning. It also extends the concept of multi-view learning to a multi-class problem setting.

### Objectives:
1. Implement the label sharing algorithm for a VFL setting with two or more partipants.
2. Experimentally validate the method.

### Improvements (Optional):
Identify 2-3 areas for improvement and incorporate these into the framework.


## Multi-VFL: A Vertical Federated Learning System for Multiple Data and Label Owners
### link: https://arxiv.org/pdf/2106.05468.pdf

### Description:
This paper considers develops a framework for jointly learning a model when the output labels are distributed across multiple parties. It incorporates a hybrid VFL-HFL architecture towards this end. VFL is employed for learning the top and bottom models initially, followed by horizontal learning over the top models. Experiments are conducted on image datasets, MNIST and FashionMNIST to validate the performance of the method. 

### Objectives:
1.. Implement private set intersection (PSI) to align the samples prior to learning.
2. Implement the framework with the hybrid learning architecture.
3. Experimentally validate the method on one image dataset and one tabular dataset of your choice.

### Improvements (Optional):
Identify 2-3 areas for improvement and incorporate these into the framework.


## Multi-Participant Vertical Federated Learning Based Time Series Prediction
### link: https://dl.acm.org/doi/pdf/10.1145/3532213.3532238

### Description:
A forecasting framework is developed for vertically-partitioned features using Gated Recurrent Units (GRUs). A split learning architecture, with top and bottom models is utilised. In addition, multi-party computation is employed to provide stronger privacy guarantees.
### Objectives:
1. Implement the framework with two versions for the top model, i.e., linear regression and GRU.
2. Experimentally validate the method on a forecasting dataset of your choice.

### Improvements (Optional):
Identify 2-3 areas for improvement and incorporate these into the framework.





