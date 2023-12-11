
:warning: 
**The first lecture starts at 9 am, Feb 19, 2024  .**
**I am still developing the course content. The content will be only be finalized by the end of the first week of the semester.**

<!-- vscode-markdown-toc -->
* 1. [Important links](#Importantlinks)
* 2. [Course description](#Coursedescription)
* 3. [Learning objective](#Objective)
* 4. [Course team](#Courseteam)
* 5. [Learning objectives](#Learningobjectives)
* 6. [:dart: Grading policy](#dart:Gradingpolicy)
	* 6.1. [Homework](#Homework)
	* 6.2. [Group projects](#Groupprojects)
* 7. [Detailed schedule](#Detailedschedule)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc --><!-- vscode-markdown-toc -->


# DDL <!-- omit in toc -->

This repository contains the materials of the **MSc on Distributed Deep Learning Systems** course running in Spring 2024 at BeNeFri.


##  1. <a name='Importantlinks'></a>Important links

- [Lectures notes](lecture.md)
- [Project description](project.md)
- [Lab and assigment](homework.md)


##  2. <a name='Coursedescription'></a>Course description


Machine learning systems are often conventionally designed for centralized processing in that they first collect data from distributed sources and then execute algorithms on a single server. Due to the limited scalability of processing large amount of data and the long latency delay, there is a strong demand for a paradigm shift to distributed or decentralized ML systems which execute ML algorithms on multiple and in some cases even geographically dispersed nodes.

The aim of this  course is to let students learn how to design and build distributed ML systems via paper reading, presentation, and discussion; We provide a broad overview on the design of the state-of-the-art distributed ML systems, with a strong focus on the scalability, resource efficiency, data requirements, and robustness of the solutions. We will present an array of methodologies and techniques that can efficiently scale ML analysis to a large number of distributed nodes against all operation conditions, e.g., system failures and malicious attacks. The specific course topics are listed below.

The course materials will be based on a mixture of classic and recently published papers. 


##  3. <a name='Objective'></a>Learning objectivs
- To argue and reason about distributed ML from a systems perspective.
- To understand the behavior and tradeoffs of distributed ML in terms of performance and scalability
- To estimate the importance of data inputs via different techniques, i.e., core set and decomposition methods, for distributed ML systems.
- To understand data poison attacks and design defense strategy for distributed ML systems.
- To analyze the state-of-the art federated machine learning systems and design the failure-resilient communication protocols
- To design and implement methods and techniques for making distributed ML systems more efficien

##  4. <a name='Courseteam'></a>Course team

This course will be mainly taught by [Prof. Lydia Y Chen](https://lydiaychen.github.io/).
TA is Abel Malan who will run the lab

Lydia is the responsible instructors of this course and can be reached at **lydiaychen@ieee.org**.


##  5. <a name='dart:Gradingpolicy'></a>:dart: Grading policy

This course has no final exam, instead the grade is largely determined through three components: 

1. Lab assigbment (30%): 3 individual lab assigment, due in week 4, 8, 11. 

2. Group project (70%): group project report (60%) and presentation (10%). The goal is to reproduce a paper and propose an algorithm to extend the paper. There will be an initial proposal in week 5, interim discussion with each team in week 9. The final report will be due in week 13, and 20 minutes presentation in week 13 as well.
   


**All assessment items (homework, and projects reports) have to be submitted via ILIAS.**


###  6.1. <a name='Homework'></a>Homework
- Homework 1: due in week 4 
- Homework 2: due in week 8
- Homework 3: due in week 11 

Students are given additional 48 hours grace period for late submission and will not receive any grade penalty. However, submissions after 48 hours grace period will not be considered and students will loose 25 points of their final grade. 


###  6.2. <a name='Groupprojects'></a>Group projects
<!-- 7 predefined project topics: evaluating the systems of 
-->
The objective is to reproduce and improve the performance of a paper from the course list (see project.md). The students need to hand in a final project report in the style of a short scientific paper, stating their individual contribution to the overall system performance. There are four milestones associated with this project (see project.md).

- Group size: 1-2 students
- Schedule: initial proposal (week 5), interim meeting (week 9), report due (week 13), and presentation/interview (week 13). 

[UPDATE] We change the requirement. For the final project, you just need to submit ppt slides, which summarize the results. If you submit a report, you will be getting bosnus point.

At the end of each project phase we will conduct a short interview (20 minutes per group) about the group project and its connection to the course content. Based on the project report and the interview, each member of the group receives a grade. 





##  7. <a name='Detailedschedule'></a>Detailed schedule

**Week**|**Topic**
:-----|:-----
Week 1 | Distributed Machine Learning I 
Week 2 | Memory and Aceeleration Technology
Week 3 | Federated Learning I (Horizontal)
Week 4 | Federated Learning II (Vertical)
Week 5 | Hyper-parameter Tuning
Week 6 | Robust Distributed Learning 
Week 7 | Privacy Enhancing Technology for Federated Learning
Week 8 | Distributed inference 
Week 9 | Self-study
Week 10| Distributed/Federated Generative AI 
Week 11| Continual Federated Learning and Domain Adaptaion
Week 12| Advanced Attacks and Defenses in Federated Learning
Week 13| Self-study
Week 14| Project presentation


