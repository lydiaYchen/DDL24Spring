
## Objectives

Your project accounts for 70% of your final grade.

Here, you need to first reproduce one of the one of the four papers below using a framework of your choice.

Secondly, you need to come up with solutions to further improve the performance metrics of your choice. Your final grade depends on (i) the rigorousness and correctness of your reproduction experiments, (ii) the percentage of performance improvement of your solution compared to the original paper, (iii) the readability of your report.

We provide you Pytorch-based of distributed/decentralized framework: You can find the implementation and user manual below. We also provide you the google cloud coupon so that you can reproduce the results on the Google cloud.

## Key milestones

Milestones:

1. **Project proposal** (mandatory, but ungraded - you will receive feedback): due on week 5
2. **Intermediate project meeting** (mandatory, but ungraded - you will receive feedback): due on week 9
3. **Final project report** (mandatory, graded): due on week 14
4. **20 min project presentation** (mandatory, graded): due on week 14

All the documents need to be submitted via the ILISAS. Exact due dates are on the ILIAS.

## Grading breakdown of the project

- Final report: 65%
- 15 min project presentation: 15 %
- Individual contribution: 20%

## Paper list

### Horizontal Federated Learning

#### Adaptive Federated Optimization

##### [Paper](https://openreview.net/pdf?id=LkFG3lB13U5) | [Code](https://github.com/google-research/federated/tree/master/optimization)

##### Description:

Proposes a federated framework supporting adaptive optimizers, which have been shown to make tuning easier and help convergence in the non-federated case. The official implementation is in Tensorflow, but the pseudocode is detailed. Experiments on various datasets cover tasks for image classification, text prediction, and image reconstruction.

##### Objectives:

- Reproduce the main results (Table 1)
- Implement 1-2 additional adaptive optimizers (e.g., Adadelta, RAdam)

##### Improvements (optional):

- Implement even more optimizers for the server
- Implement other client optimizers than SGD
- etc.

#### Personalized Federated Learning with Moreau Envelopes

##### [Paper](https://arxiv.org/pdf/2006.08848.pdf) | [Code](https://github.com/CharlieDinh/pFedMe)

##### Description:

Formulates a new way to personalize client models, which helps them perform better on their local tasks than using an unmodified version of the global one. It uses Moreau Envelopes, a mathematical function that yields an approximate version of the input function as a regularization term. It uses one real dataset in addition to synthetic ones.

##### Objectives:

- Reproduce the main MNIST results (Figure 6 & Table 1)
- Investigate the results of applying one of the attacks from the lab

##### Improvements (optional):

- Experiment with additional real datasets
- Add a defense against the investigated attack
- etc.

#### Data-Free Knowledge Distillation for Heterogeneous Federated Learning

##### [Paper](https://arxiv.org/abs/2105.10056) | [Code](https://github.com/zhuangdizhu/FedGen)

##### Description:

Devises a procedure for applying knowledge distillation in federated learning to address challenges created by user heterogeneity without needing a proxy dataset. Assuming client models compose an extractor (to bring features into latent space) and a predictor, the server shares with clients a latent sample generator that incorporates information from all of them to aid with data augmentation during local training.

##### Objectives:

- Reproduce the results of FedAvg & FedGen for MNIST & MNIST (from Table 1)
- Investigate the effect of other generator architectures than the two-layer MLP

##### Improvements (optional):

- Reproduce 1-2 additional baseline methods or datasets
- Check the impact of coupling the generator sample count to the local client dataset size
- etc.

### Vertical Federated Learning

#### Multi-Participant Multi-Class Vertical Federated Learning

##### [Paper](https://arxiv.org/pdf/2001.11154.pdf)

##### Description:

Typically, vertical federated learning (VFL) involves one party possessing the labels with the features being partitioned across multiple participants. This paper provides a privacy-preserving label-sharing algorithm to enable joint learning. It also extends the concept of multi-view learning to a multi-class problem setting.

##### Objectives:

1. Implement the label sharing algorithm for a VFL setting with two or more partipants.
2. Experimentally validate the method.

##### Improvements (optional):

Identify 2-3 areas for improvement and incorporate these into the framework.

#### Multi-VFL: A Vertical Federated Learning System for Multiple Data and Label Owners

##### [Paper](https://arxiv.org/pdf/2106.05468.pdf)

##### Description:

This paper considers develops a framework for jointly learning a model when the output labels are distributed across multiple parties. It incorporates a hybrid VFL-HFL architecture towards this end. VFL is employed for learning the top and bottom models initially, followed by horizontal learning over the top models. Experiments are conducted on image datasets, MNIST and FashionMNIST to validate the performance of the method.

##### Objectives:

1.. Implement private set intersection (PSI) to align the samples prior to learning.
2. Implement the framework with the hybrid learning architecture.
3. Experimentally validate the method on one image dataset and one tabular dataset of your choice.

##### Improvements (optional):

Identify 2-3 areas for improvement and incorporate these into the framework.

#### Multi-Participant Vertical Federated Learning Based Time Series Prediction

##### [Paper](https://dl.acm.org/doi/pdf/10.1145/3532213.3532238)

##### Description:

A forecasting framework is developed for vertically-partitioned features using Gated Recurrent Units (GRUs). A split learning architecture, with top and bottom models is utilised. In addition, multi-party computation is employed to provide stronger privacy guarantees.

##### Objectives:

1. Implement the framework with two versions for the top model, i.e., linear regression and GRU.
2. Experimentally validate the method on a forecasting dataset of your choice.

##### Improvements (optional):

Identify 2-3 areas for improvement and incorporate these into the framework.

## Report formats

### Project proposal

The proposal should contain between 300 and 400 words. Your proposal should address the following points:

1. Which paper to reproduce
2. What configuration
3. Which figures and tables

**Submission**: every group uploads their proposal on ILIAS. The proposal should be in PDF format and should contain the group name and the list of group members (name, student IDs).

**Feedback**: the course team will provide feedback on the project proposal within a few days.

### Final project report

The **final** group project report should be **20-25 slides**.
 <!---The **intermediate** project report is likely to be shorter (it is due a week before the final deadline), that is fine, submit whatever you have by then.)
 -->

We suggest the following final report structure:

- Motivation
- Background of the paper
- Method proposed by the paper
- Reproducing results (with a lot of figures and tables)
- Your proposed improvement, describing the weakness of current paper and  your optimization algorithms
- Conclusions: what can be the future direction
- References

The final report should also contain a link to a repository (or several) that contain the software you created, the scripts you used to analyze your data, etc.

Submission: every group uploads their intermediate presentation and final group project reports on ILIAS. The reports should be in PDF format.

In addition, your report must also include a figure that graphically depicts a major component of your project (e.g., your approach and how it relates to the application, etc.). Such a summary figure makes your paper much more accessible by providing a visual counterpart to the text. Developing such a concise and clear figure can actually be quite time-consuming; I often go through around ten versions before I end up with a good final version.

The final project report is graded in combination with the project interview. The interview will be an academic discussion about the executed project.

## Interviews

The 15 minute interviews per group will be scheduled on the last lecture.
