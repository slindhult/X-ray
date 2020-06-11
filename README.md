# Computer Assisted Diagnosis

<p float="left">
  <img src='https://www.alfredhealth.org.au/images/made/contents/general/Stock-images-owned/Xray-banner_1200_400.jpg'  />
</p>
# Table of Contents
1. [Introduction](https://github.com/slindhult/X-ray#Why-Chest-X-rays)
2. [Models](#example2)
3. [Third Example](#third-example)
4. [Further Considerations](#fourth-examplehttpwwwfourthexamplecom)

### Why Chest X-rays
Chest X-rays are one of the most common and cost effective medical imaging examinations available for chest ailments, however clinical diagnosis of a chest x-ray can be challenging and sometimes more difficult than diagnosis via CT, which may not be available at rural medical centers. 

With computer aided diagnostics provided through predictive models using convolutional neural networks.  Health care facilities could provide better predictions by providing busy doctors with suggested pathologies to look into with their associated probabilities.  More accurate diagnosis could lead to better care especially if no other diagnostic equipment is available.

### The Dataset
The dataset was pulled from [Kaggle](https://www.kaggle.com/nih-chest-xrays/data) and provided by the National Institute for Health and contained 112k chest X-ray images with disease labels of 14 potential pathologies or ‘No Finding’.  It also included a csv file containing the associated patient age, gender,view of the image, patient number and followup appointment number. 
60,000 No Finding
60,000 No Finding
27,000 Single Pathology X-rays
24,600 Multiple pathology X-rays
Over 800 different combinations for the diagnosis

#### Pathologies: 16 Most common Diagnoses
![Top 16](https://github.com/slindhult/X-ray/blob/master/figures/diagnoses_master.jpg?raw=true)


The dataset was evenly distributed between male and female across age range, pathology, and the view the image was taken from Anterior-Posterior (AP) or Posterior-Anterion(PA).  Since it is evenly distributed a model trained on the entire dataset should generalize well.
<p float="left">
  <img src="https://github.com/slindhult/X-ray/blob/master/figures/agegender.png?raw=true" width="400" />
  <img src="https://github.com/slindhult/X-ray/blob/master/figures/mfdistribution.png?raw=true" width="400" /> 
</p>

#### Initial Model
Results: 
* 65%  using Top 3 accuracy
* 45% with a custom accuracy score incorporating a penalty component.
The penalty component was implemented because not diagnosing the patient with something they do have is a major problem in health care and would lead to a delay in care.  
![Custom Scoring](https://github.com/slindhult/X-ray/blob/master/figures/custom_scoring.png?raw=true)

#### Simplifying the problem
Diagnosis using the full dataset was a very difficult problem as it would be a 15 class multi-classification problem with a range of  output sizes.  To have more success a pivot was necessary to break the problem down into smaller pieces.

The problem was broken down into smaller individual diagnosis models that were binary classifiers comparing a positive finding against the no finding category in a more patient specific way.

#### Model Architecture
The model used four conv2D layers with a Relu activation function to loop through the pixels and extract features.  The model also utilized four maxpooling layers with filter size 2x2 which looped through the pixels and selected the maximium value in the filter, leading to a reduced image size after each layer.  

!Model Architecture](https://github.com/slindhult/X-ray/blob/master/figures/model_architecture.png?raw=true)

#### Simplified Models
The simplified models compared individual findings against the no finding category.  This required creating new datasets for each pathology combined with the no finding data and balancing it, since the classes would be very imbalanced.

##### Atelectasis
Below are the results of the model on Atelectasis, the partial collapse of the lung where the alveoli fill with alveolar fluid.  The accuracy was 81.6%, but important metric is the specificity(recall) which was 88.8%.  Specificity is the important metric because a prediction of healthy when someone is unhealth would be a major issue, it's better err on the side of caution and maximize recall over precision or accuracy. 

<p float="left">
  <img src="https://github.com/slindhult/X-ray/blob/master/figures/Atelectasis_confusion.png?raw=true" width="400" />
  <img src="https://github.com/slindhult/X-ray/blob/master/figures/atelectasis_example.png?raw=true2" width="400" /> 
72

</p>

##### Effusion
Below are the results of the model on Effusion, the build up of fluid in the tissue aroudn the lung collapse of the lung. The accuracy was 81.6%, but important metric is the specificity(recall) which was 71.59%. Specificity is the important metric because a prediction of healthy when someone is unhealth would be a major issue, it's better err on the side of caution and maximize recall over precision or accuracy.

<p float="left">
  <img src="https://github.com/slindhult/X-ray/blob/master/figures/effusion_confusion.png?raw=true" width="400" />
  <img src="https://github.com/slindhult/X-ray/blob/master/figures/effusion_example.png?raw=true" width="400" /> 
</p>


### Future Considerations:
* Introduce a cutoff point for each pathology based on the model to create a more appropriate screening buffer and drive recall to near 100%.
* Further training of the models on larger images to see if diagnosis improves as the images were significantly reduced in size for to reduce training time.
