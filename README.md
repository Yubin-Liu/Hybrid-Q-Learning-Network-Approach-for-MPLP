# Hybrid-Q-Learning-Network-Approach-for-MPLP

This code contributes to the research entitled with "Route Planning for Last-Mile Deliveries Using Mobile Parcel Lockers: A Hybrid Q-Learning Network Approach" that submits to Transportation Research Part E: Transporation and Logistics Review (ID:TRE-D-23-00202). The following is an abstrat of the research:   

Mobile parcel lockers (MPLs) have been recently proposed by logistics operators as a technology that could help reduce traffic congestion and operational costs in urban freight distribution. Given their ability to relocate throughout their area of deployment, they hold the potential to improve customer accessibility and convenience. In this study, we formulate the Mobile Parcel Locker Problem (MPLP), a special case of the Location-Routing Problem (LRP) which determines the optimal stopover location for MPLs throughout the day and plans corresponding delivery routes. A Hybrid Q-Learning-Network-based Method (HQM) is developed to resolve the computational complexity of the resulting large problem instances while escaping local optima. In addition, the HQM is integrated with global and local search mechanisms to resolve the dilemma of exploration and exploitation faced by classic reinforcement learning (RL) methods. We examine the performance of HQM under different problem sizes (up to 200 nodes) and benchmarked it against the exact approach and Genetic Algorithm (GA). Our results indicate that the average reward obtained by HQM is 1.96 times greater than GA’s, which demonstrates that HQM has a better optimisation ability. Further, we identify critical factors that contribute to fleet size requirements, travel distances, and service delays. Our findings outline that the efficiency of MPLs is mainly contingent on the length of time windows and the deployment of MPL stopovers. Finally, we highlight managerial implications based on parametric analysis to provide guidance for logistics operators in the context of efficient last-mile distribution operations.

The code consists of the following parts:

generate_demand.py---To generate the location and demand randomly. The location are generated via K-means clustering.

genetic_algorithm.py---To implement GA for the performance comparison.

HQM.py---To implement the Hybrid Q-Learning Network Approach. The mechanism of HQM see the research article.

main.py---To implement all related .py file of HQM and GA. Output and visualise the implementation results.

result.py---Include all result print functions and plot functions.

reward_GA.py---Define the reward of GA. This reward is recalled by genetic_algorithm.py.

reward_HQM.py---Define the reward of HQM. This reward is recalled by HQM.py.

The result folders provides a sample result for a large problem instance for reference (the number of parking space = 10, the number of customer locations within each parking space = 20).
