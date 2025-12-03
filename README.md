# Threat Classification System

<div align="center"> 
<img src="img/1.png" alt="Threat Detection Logo", >
</div>
<br> 

<div align="justify"> 
This project centers around the development of an intelligent threat detection system designed to assist blind and visually impaired individuals in navigating their surroundings safely. Built using deep learning-based computer vision techniques, the system identifies potential environmental threats—such as moving vehicles, static obstacles, and drop-offs—and alerts users through real-time auditory feedback. </div><br> 

<div align="justify"> 
The core of this solution leverages convolutional neural networks (CNNs), trained on a custom-curated Threat Object Detection (TOD) dataset comprising over 100,000 images. The model is optimized using techniques such as transfer learning and data augmentation to ensure high accuracy and generalization across dynamic environments. This system is intended for use with wearable cameras or smartphone-integrated devices, making it both accessible and scalable for real-world deployment. </div><br> 

<div align="justify"> The ultimate goal is to enhance the independence, confidence, and safety of blind individuals by bridging the gap between traditional assistive tools and intelligent visual perception. The system has been evaluated through user studies and demonstrates significant promise as a next-generation assistive technology solution. </div>

## Setup
<h3>
<div>
<img src="img/env.png"; width=20> <b>Environment</b>
</div>
</h3>

*Note - The above codes are written using Python virtual environment version **3.12.1** and supports the latest Python versions. However, lower Python versions may offer more stability compared to the latest ones. Below setup will focus Windows system. And commands to setup may vary for macOS or Linux.*



<div> 1. Move to folder (CLI commands)
```bash
# If you downloaded the file, just navigate to folder
# Then press Shift (in Windows) + Left Mouse click
# Choose Open PowerShell window here option and move on to creating virtual environment
cd threat_object_nn
```
</div>

<div> 2. Create a python virtual environment using -
```bash
# Path to installation of particular version of python  may vary
# I have installed more than one version of python in pyver directory
# Refer the resources section, for youtube video to install multiple versions of python
C:\Users\<username>\pyver\py3121\python -m venv threatenv
```
</div>
  
<div> 3. Activate the virtual environment using -
```bash
threatenv\Scripts\activate
```
</div>

<div> 4. Install python packages using - 
```bash
pip install -r requirements.txt
```
</div>

<div> 5. Jupyter Notebook Configuration -

```bash
ipython kernel install --user --name=myenv
python -m ipykernel install --user --name=myenv
```
</div>

<div> 6. For bash based cell execution inside Jupyter (Optional) -

```bash
pip install bash_kernel
python -m bash_kernel.install
```
</div>

<div> 7. Congratulations! Now the environment is ready for execution -

```bash
jupyter notebook
```
</div>

## CNN Architecture

<div align="justify">
<div align="center"> 
<img src="img/2.png" alt="Threat Detection Logo"><br><i>Schematic of Convolution neural network</i>
</div>

<br> 
This project leverages Convolutional Neural Networks (CNNs) for threat detection through image classification. CNNs are particularly effective for visual data tasks due to their ability to automatically learn spatial hierarchies of features. Each stage of the CNN architecture is carefully designed to extract features and classify input images with high accuracy.
</div><br>

<div align="center"> 
<img src="img/3.png" alt="Threat Detection Logo"><br><i>CNN Model Architecture</i> 
</div>

**Convolutional Layers (Conv2D)**
<ul>
<li> These layers apply learnable filters (kernels) to the input image, generating feature maps that capture patterns such as edges, corners, or textures.</li>
<li>Purpose: Feature extraction</li>
<li>Activation Function: ReLU (Rectified Linear Unit) introduces non-linearity</li>
<li>Example: `Conv2D(filters=32, kernel_size=(3,3), activation='relu')`</li>
</ul><br>

**Pooling Layers (MaxPool2D)**
<ul>
<li>These layers reduce the spatial dimensions of the feature maps. Pooling makes the network more robust and reduces computation.</li>
<li>Type Used: Max Pooling</li>
<li>Purpose: Downsampling and noise suppression</li>
<li>Example: `MaxPool2D(pool_size=(2, 2))`</li>
</ul><br>


**Flattening Layer**
<ul>
<li>Converts the final pooled feature maps into a one-dimensional vector. This forms the input to the dense (fully connected) layers.</li>
<li>Purpose: Transition from feature extraction to classification</li>
</ul><br>

**Dense Layers (Fully Connected)**
Dense layers make predictions based on the extracted features. In this project:
<ul>
<li>First Dense Layer: 512 neurons, ReLU activation – captures high-level features</li>
<li>Final Dense Layer: 1 neuron, sigmoid activation – outputs probability for binary classification</li>
</ul><br>

**Loss Function and Optimizer**
<ul>
<li>Loss Function: binary_crossentropy – suitable for binary (threat / no-threat) problems</li>
<li>Optimizer: RMSprop with a learning rate of 1e-4 – adjusts learning efficiently</li>
</ul>

## Results
<div align="justify">
The model was trained for 100 epochs, and performance was monitored using validation accuracy and validation loss. Below is a summarized table showing the best metrics per group of 25 epochs:
</div><br>

| Epoch Range | Best Val Accuracy | Min Val Loss | Corresponding Epoch | Notes |
|:--:|:--:|:--:|:--:|:--:|
| 1–25 | 0.9250 | 0.3228 | Epoch 21 | Accuracy improved from 48% to 92.5%, early convergence observed |
| 26–50 | 0.9125 | 0.2486 | Epoch 50 | Strong performance across multiple epochs (>87% accuracy) |
| 51–75 | 0.9750 | 0.1008 | Epoch 68 | Peak accuracy and lowest loss achieved, model well-tuned |
| 76–100 | 0.9500 | 0.1030 | Epoch 86| Performance remained stable and generalization retained |
<br>

<h3>Visualization</h3>
<div align="center"> 
<img src="img/4.png"; alt="Result"; width=700; height=350><br><i>Visualization of accuracy for training and validation </i> 
</div><br>

<div align="center"> 
<img src="img/5.png"; alt="Result"; width=700; height=350><br><i>Visualization of loss from training and validation</i> 
</div>

## References
<ol>
<li>World Health Organization. (2021). Blindness and vision impairment. 
https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment </li>
<li>Hersh, M. A.,&Johnson, M. A. (2008). Assistive technology for visually impaired and blind people. 
Springer Science & Business Media. </li>
<li>Zhao, Z. Q., Zheng, P., Xu, S. T., & Wu, X. (2019). Object detection with deep learning: A review. IEEE 
transactions on neural networks and learning systems, 30(11), 3212-3232. </li>
<li>Manduchi, R.,&Kurniawan, S. (2011). Mobility-related accidents experienced by people with visual 
impairment. AER Journal: Research and Practice in Visual Impairment and Blindness, 4(2), 44-54. </li>
<li>Blasch, B. B., Wiener, W. R.,&Welsh, R. L. (1997). Foundations of orientation and mobility (2nd ed.). 
AFB Press. </li>
<li>Whitmarsh, L. (2005). The benefits of guide dog ownership. Visual Impairment Research, 7(1), 27-42. </li>
<li>Mandal, S.,&Chia, S. C. (2019).Areview of assistive technology for visually impaired people. In 
Proceedings of the 2019 5th International Conference on Computing and Artificial Intelligence (pp. 47-52). </li>
<li>Elmannai, W., & Elleithy, K. (2017). Sensor-based assistive devices for visually-impaired people: Current 
status, challenges, and future directions. Sensors, 17(3), 565. </li>
<li>National Eye Institute. (2021). Low Vision and Blindness Rehabilitation. https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/low-vision</li>
<li>SWANProject. (2021). System for Wearable Audio Navigation. https://www.swan-project.eu/ </li>
<li>Sosa-García, J.,&Odone, F. (2017). "Hands On" visual recognition for visually impaired users. ACM 
Transactions on Accessible Computing (TACCESS), 10(3), 1-30. </li>
<li>Hicks, S. L., Wilson, I., Muhammed, L., Worsfold, J., Downes, S. M., & Kennard, C. (2013). A 
depth-based head-mounted visual display to aid navigation in partially sighted individuals. PLoS One, 8(7), 
e67695. </li>
<li>Lakde, C. K.,&Prasad, P. S. (2015). Navigation system for visually impaired people. In Proceedings of the 
2015 International Conference on Computation of Power, Energy, Information and Communication 
(ICCPEIC) (pp. 0093-0098). IEEE. </li>
<li>Real, S.,&Araujo, A. (2019). Navigation systems for the blind and visually impaired: Past work, 
challenges, and open problems. Sensors, 19(15), 3404. </li>
<li>Elmannai, W., & Elleithy, K. (2017).Ahighly accurate and reliable data fusion framework for guiding the 
visually impaired. IEEE Access, 6, 33029-33054. </li>
<li>Lin, B. S., Lee, C. C.,&Chiang, P. Y. (2017). Simple smartphone-based guiding system for visually 
impaired people. Sensors, 17(6), 1371. </li>
<li>Poggi, M.,&Mattoccia, S. (2016). wearable mobility aid blind people: visual-inertial obstacle detection 
learning-based terrain classification. IEEE Transactions Human-Machine Systems, 46(4), 454-471. </li>
<li>Lin, B. S., Lee, C. C.,&Chiang, P. Y. (2017). Simple smartphone-based guiding system visually impaired 
people. Sensors, 17(6), 1371. </li>
<li>Yang, K., Wang, K., Hu, W., & Bai, J. (2016). Expanding traversable region visually impaired with RGB-D 
sensor. Sensors, 16(11), 1954. </li>
<li>Takizawa, H., Yamaguchi, S., Aoyagi, M., Ezaki, N., & Mizuno, S. (2015). Kinect cane: object recognition 
aids visually impaired people in new places. IEEE Transactions Computational Intelligence AI in Games, 
7(4), 320-328. </li>
<li>Katzschmann, R. K., Araki, B.,&Rus, D. (2018). Safe local navigation visually impaired users with 
time-of-flight depth camera wearable haptic feedback. IEEE Transactions Neural Systems Rehabilitation 
Engineering, 26(3), 583-593.</li>
<li>Cheng, R., Wang, K., Yang, K., Long, N., Hu, W., Chen, H., ... & Bai, J. (2017). Robust obstacle detection 
visually impaired using deformable grid pattern depth camera. IEEE Transactions Industrial Electronics, 
65(4), 3223-3233. </li>
<li>Tang, J., Cheng, J., Huang, Y., & Liu, Y. (2018). Traversable area stairs detection visually impaired based 
RGB-D camera. Sensors, 18(8), 2435. </li>
<li>Bhowmick, A., Prakash, C., Hazarika, S. M., & Raju, P. S. (2017). Wearable navigation assistance system 
visually impaired based depth sensor haptic feedback. IEEE Transactions Haptics, 11(1), 71-81. </li>
<li>Lin, Y. H., Wang, K., Yi, W., & Lian, S. (2019). Real-time obstacle detection avoidance visually impaired 
using binocular camera. IEEE Access, 7, 23272-23282. </li>
<li>Xiao, L., Zhu, Y., Shi, Y., Mao, X.,&Shi, Y. (2018). Autonomous navigation system visually impaired 
using wearable RGB-D camera vibrotactile feedback. IEEE Sensors Journal, 18(22), 9253-9263. </li>
<li>Lin, B. S., Chiang, P. Y., & Lee, C. C. (2020). Wearable assistive device visually impaired integrating depth 
sensing semantic segmentation. IEEE Sensors Journal, 20(18), 10711-10720. </li>
<li>Kaushalya, V., Premarathne, K., Shadir, H., Krithika, P., & Fernando, P. (2019). Obstacle avoidance system 
visually impaired based ultrasonic sensors vibrotactile feedback. International Journal Advanced Computer 
Science Applications, 10(5), 374-380. </li>
<li>Yang, K., Wang, K., Cheng, R., Hu, W., Huang, X., & Bai, J. (2017). deep learning-based obstacle 
detection avoidance visually impaired using RGB-D camera. Journal Sensors, 2017, 1-11. </li>
<li>Bai, J., Liu, Z., Lin, Y., Li, Y., Lian, S., & Liu, D. (2019). Wearable assistive system visually impaired 
integrating haptic feedback auditory cues. IEEE Transactions Industrial Informatics, 16(3), 1463-1473. </li>
<li>Cheng, R., Wang, K., Lin, L.,&Hu, W. (2019). Traversable area detection visually impaired using RGB-D 
camera polarization cues. Optics Express, 27(4), 4399-4413.</li>
</ol>
