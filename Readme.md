<script src="//yihui.org/js/math-code.js"></script>
<!-- Just one possible MathJax CDN below. You may use others. -->
<script async
  src="//mathjax.rstudio.com/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

# Introduction
In this project, I’m trying to develop an algorithm that enables a spaceship to most efficiently travel through a gravitational field. Suppose our spaceship starts from a point A and wants to travel to point B. It is known that gravitational objects like stars or planets reside in the space between A and B, but we are not exactly sure about their status or it is too complex to analyze their motions, i.e., we don’t have a well-defined model. Therefore, it may be an interesting problem to design a model-free algorithm that enables the spaceship to autonomously decide its actions based on the feedback from the environment, i.e. forces experienced by the spaceship, and its status, ie. its relative location to B, velocity, and acceleration etc. The project tries to solve this problem based on Q-Learning, a model-free reinforcement learning algorithm, which will be described more in details in the next section. For the sake of simplifying the problem, I assume a 2D space and use point estimates for the spaceship and other gravitational objects, so I ignore all the angular motions. Further, I assume that the spaceship can only accelerate in directions that are perpendicular to AB and that the magnitude of acceleration is constant, so there are only two plausible actions for the spaceship. I also have not considered the usage of fuel, i.e., the spaceship can make infinite propulsions.

# Background (Q-Learning)
Problem Formulation
Assume we start from the origin A, and the target is B, which is on the x-axis.
The set of states S, `$(r_x, r_y, v_x, v_y)$`
The set of actions A = {[0,\delta v],[0,-\delta v],[0,0]}, which represents 1)add a constant value to v_y (accelerate in the y direction), 2)substract a constant value from v_y (decelerate in the y direction), and 3) no acceleration 
The transition function, based on previous state s_t, use N-body solver to calculate the accelerations and evolve all positions and velocities to derive s_{t+1}
The reward funciton, R(s_t, s_{t+1}) = C(||r_{t} - B||_2 - ||r_{t+1} - B||_2), where C = \frac{1}{||A - B||}, which measures how well does the current step perform comparing to the previous state, i.e. Did the action brings the spaceship closer to B.

# Implementation
To implement the algorithm, I utilize object-oriented programming feature of Python. I created two classes, 

# Results

# Discussion and Further Work
I think this project is intersting in many ways. Certianly human beings would not be able to travel with spaceship through long distance in the near future, the algorihtm may be applied to other robotic systems like autonomous vehicles. (They are of course using much more sophisticated and accurate techniques.)
As I claimed in the introduction, I make several limited assumptions and put several limits on the spaceship. It's possible to break those limits with some further work including:

1. Extend 2D stimulation to 3D;
2. Enable the spaceship to complete more complicated actions such as the ability to accelerate in more directions;
3. Take into account the use of fuel, for example penalizing the model if it makes too many accelerations;
4. Take into account the angular motion. 