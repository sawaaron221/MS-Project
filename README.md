<h1> Developing Convolutional Neural Network Models to Mitigate Computation Time for Computational Fluid Dynamics </h1>
<h2> Description </h2>
The project developed two deep convolutional neural networks to upscale low- fidelity CFD to high- fidelity CFD simulations. The study focused on two-dimensional flows with five different geometries under subsonic flow conditions. ANSYS Fluent was used to generate the coarse dataset and the fine dataset to use as the inputs and the ground truths, respectively. The models were built using convolutional neural networks with regression algorithms.
<br/>

![image](https://user-images.githubusercontent.com/130534007/233774850-413bcace-890f-4c6b-a08e-2a2929c463be.png)

![image](https://user-images.githubusercontent.com/130534007/233774851-acfea444-7f11-4550-b921-2a2c8b09e8ca.png)

<br/>

Table 3.1 – 1st Model: Concatenated Convolutional Neural Network
	Output Shape
Channels/filter x height x width	Dropout %	Number of Parameters
Input variables	3 x 256 x 256	-	-
Concatenate Layer	9 x 256 x 256	-	-
Convolution Block	128 x 256 x 256	10	158.6 k
Convolution Block	256 x 128 x 128	10	886.2 k
Convolution Block	512 x 64 x 64	15	3.5 M
Convolution Block	1024 x 32 x 32	15	14.1 M
Convolution Block	1024 x 16 x 16	20	18.8 M
Convolution Block	1024 x 8 x 8	20	18.8 M
Convolution Block	2048 x 4 x 4	25	56.6 M
Center Convolution Block	2048 x 2 x 2	25	75.5 M
Transposed Convolution Block	2048 x 4 x 4	25	130.0 M
Transposed Convolution Block	1024 x 8 x 8	20	36.7 M
Transposed Convolution Block	1024 x 16 x 16	20	32.5 M
Transposed Convolution Block	1024 x 32 x 32	15	32.5 M
Transposed Convolution Block	512 x 64 x 64	15	9.1 M
Transposed Convolution Block	256 x 128 x 128	10	2.3 M
Transposed Convolution Block	128 x 256 x 256	10	574.3 k
Transposed Convolution Layer		3 x 256 x 256		(387) x 3
Total Trainable Parameters			432.4 M
Non-trainable Parameters			28.1 k
