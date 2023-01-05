# SobolevTraining

This project is an example of Sobolev training, where you enforce a neural networks gradients to match your data in addition to normal supervised learning.
Usually, NNs are trained using X,Y data pairs where the NN is forced to output Y given X. Sobolev training additionally provides X,Y,dY/dX pairs and forces
the NN's gradients to match the data. As a result, the NN better approximates the function with less data since the set of all possible approximations it can choose
is more limited. See the below images for examples of training 10 NNs for both training methods on different curves. Feel free to use this code for reference. 

![image](https://user-images.githubusercontent.com/105821676/210645189-b8f915f7-d185-4743-b8fc-cc59827e42ca.png)

![image](https://user-images.githubusercontent.com/105821676/210646428-05c578aa-1fb5-4df6-951e-a9a301ed9e66.png)