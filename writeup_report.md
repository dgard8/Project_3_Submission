# **Behavioral Cloning** 

## Writeup Template


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[centerImage]: ./images/center_image.jpg "Center Image"
[leftImage]: ./images/left_image.jpg "Left Image"
[rightImage]: ./images/right_image.jpg "Right Image"
[recovery1]: ./images/recovery_1.jpg "Recovery 1"
[recovery2]: ./images/recovery_2.jpg "Recovery 2"
[recovery3]: ./images/recovery_3.jpg "Recovery 3"

---
### Code Summary

Code for the nueral network is contained in model.py. The program accepts a list of folders as command line arguments. The network will train on the images in those folders. I stored data for different scenarios in different folders (one for normal driving, one for turns, one for error recovery, etc), so I could see how including or excluding certain types of data affected the result.

The program also has two flags. One to load the existing model and fine-tune it instead of retraining from scratch. The other to change the number of epochs (default is two).

### Model Architecture

I am using the model architecture that Nvidia uses in their published solution, which consists of five convolution layers and four fully connected layers. I had originally tested with LeNet, but found that it couldn't get past the first turn. Switching to Nvidia's significantly improved performance.

I pre-process the images by cropping off the background noise of the treeline and car bumper. Then I do the standard (x-255)/0.5 to normalize the pixels.

The model uses dropout after each layer to reduce overfitting. I also split the incoming data into a training and a validation set to compare the loss on each one. The loss was similar in each set which implies the network isn't overfitting.

I used the adam optimizer option in Keras so I didn't specify a learning rate.

### Testing Data

I found it was difficult to generate high quality data using the keyboard keys. The mouse was slightly better but still not great. As such I relied on the data provided in the project resources for the base of my testing data. After training with that data I found places where the car would leave the road (the dirt path and the right turn were the biggest problems) and took my own data in those places to help the model learn how to handle them.

To help the system learn how to handle off-center driving, I used the images from the left and right cameras. I found that a correction of 0.05 helped the model perform the best.

#### Data Images

The majority of the testing data is normal center-lane driving. This is important because we want the car to drive in the center, so we need our data to teach it how to do that. Here is an image of what driving in the center looks like:

![center driving][centerImage]

To ensure the car would move back to the center if it drifted to the side of the road, I recorded data starting at the side of the road and moving back towards the center. Here are some images that show what this looks like and what we hope the car will do if it drifts (though ideally it never gets off-center in the first place):

![recovery starting][recovery1]
![recovery middle][recovery2]
![recovery end][recovery3]

To further help the car move toward the center and to augment the data, I used the cameras on the right and left side of the car. I added/subtraced 0.05 from the steering angle so the model would learn that it needs to turn more/less if it gets off center. Here are what the three images from the car look like:

![center driving][centerImage]
![left driving][leftImage]
![right driving][rightImage]

### Conclusion

I was able to achieve fairly successful autonomous driving. The car does get pretty close to the edge right around the dirt turn-off but it manages to recover successfully. This was acheived using the provided data supplemented with data taken around turns and data taken recovering from the edge of the road.

