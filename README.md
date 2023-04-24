<h1> Developing Convolutional Neural Network Models to Mitigate Computation Time for Computational Fluid Dynamics </h1>
<h2> Description </h2>
The project developed two deep convolutional neural networks (CNN) to upscale low- fidelity computational fluid dynamics (CFD) to high- fidelity CFD simulations. The study focused on two-dimensional flows with five different airfoils geometries under subsonic flow conditions. The coarse and fine datasets, used as the inputs and the ground truths, respectively, were generated using ANSYS Fluent. The machine learning models were built using Python and TensorFlow.
<br/>
<br/>
<h2> Mesh: Shows an example mesh of one of the five airfoils. </h2>
<p align="center">
<img src="https://user-images.githubusercontent.com/130534007/233774850-413bcace-890f-4c6b-a08e-2a2929c463be.png" height="60%" width="60%"/>
<img src="https://user-images.githubusercontent.com/130534007/233774851-acfea444-7f11-4550-b921-2a2c8b09e8ca.png" height="60%" width="60%"/>
<br/>
Figure 1. This is the fine-grid mesh for NACA-6409 airfoil. The mesh has a total of 368,872 elements.  
<br/>
<br/>
Table 1 – Number of elements for coarse and fine meshes.  <br/>
<img src="https://user-images.githubusercontent.com/130534007/233892467-c58bf97c-cf30-4f6f-b20d-bab3e0b9947b.png" height="80%" width="80%"/>
<br/>
</p>
<h2>Machine Learning Models: Shows the total parameters of each model and diagrams of the CNN architecture.</h2>
<p align="center">
Table 2 – 1st Model: Concatenated Convolutional Neural Network
<img src="https://user-images.githubusercontent.com/130534007/233892972-1f6e557f-e54b-4196-8928-f6187fbe64ca.png" height="80%" width="80%"/>
<br/>
<br/>
<img src="https://user-images.githubusercontent.com/130534007/233893110-025005e1-cfbd-4a1f-9c6d-1a202a00e391.png" height="70%" width="70%"/>
<br/>
Figure 2. – Concatenated convolutional neural network architecture (1st model). The number of filters/channels is denoted next to the box, highlighted in “black.” The pixel dimensions are denoted inside the box, highlighted in “white.” 
<br/>
<br/>
Table 3 – 2nd Model: Split Convolutional Neural Network
<img src="https://user-images.githubusercontent.com/130534007/233893715-2273c100-4612-4cd3-b217-b1e50149e4e5.png" height="80%" width="80%"/>
<br/>
<br/>
<img src="https://user-images.githubusercontent.com/130534007/233893917-7cdf23e8-22cf-4c04-932b-3f376f32e171.png" height="70%" width="70%"/>
<br/>
Figure 3 – Split convolutional neural network architecture (2nd model). The number of filters/channels is denoted next to the box, highlighted in “black.” The pixel dimensions are denoted inside the box, highlighted in “white.” 
</p>
<h2>Results: </h2>
<p align="center">
<img src="https://user-images.githubusercontent.com/130534007/233895142-7f986a91-1b96-43b4-8f12-55a65a756d21.png" height="70%" width="70%"/>
<br/>
Figure 4 – NACA-6409 velocity contour at a high angle of attack, α>35^o. The top images are the coarse mesh simulation results (inputs), the middle images are the machine learning models’ predictions, and the bottom images are the fine mesh simulation results. The left side represents the split model, and the right side represents the concatenated model.
<br/>
<br/>
<img src="https://user-images.githubusercontent.com/130534007/233895278-4b6a2807-4749-46a6-85b6-c7d1fc03d8d3.png" height="70%" width="70%"/>
<br/>
Figure 5 – This image represents NACA 6409 airfoil at a high angle of attack, α>35^o. The difference between the ground truth and the predictions from the models are highlighted with the white square. 
</p>
<h2>Conclusion: </h2>
