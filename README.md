# SobolevTraining

This project is an example of Sobolev training, where you enforce a neural networks gradients to match your data in addition to normal supervised learning.
Usually, NNs are trained using X,Y data pairs where the NN is forced to output Y given X. Sobolev training additionally provides X,Y,dY/dX pairs and forces
the NN's gradients to match the data. As a result, the NN better approximates the function with less data since the set of all possible approximations it can choose
is more limited. See the below images for examples of training 10 NNs for both training methods on different curves. Feel free to use this code for reference. 

![image](https://user-images.githubusercontent.com/105821676/210645189-b8f915f7-d185-4743-b8fc-cc59827e42ca.png)

![image](https://user-images.githubusercontent.com/105821676/210646428-05c578aa-1fb5-4df6-951e-a9a301ed9e66.png)

Additionally, we can render different approximations throughout training. The following video shows typical batched supervised training, where the batch is not large enough to represent the entire range of inputs. This is quite common, especially as the number of inputs increases. Imagine how large a batch would have to be for input data with 100s, 1,000s, or 1,000,000s of inputs. Therefore, I expect this is a frequent occurence in real life applications.

The normal training method is noticably jittery during training. It takes far longer to converge, and occasionally "forgets" a part of the function which is not represented in the current batch. In contrast, Sobolev training is noticably less jittery and converges quickly to the true function. 





https://user-images.githubusercontent.com/105821676/211069288-81b00e17-8f95-4f34-8464-88769382fcdc.mp4

