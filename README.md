# TASK-3  MLOPS
# INTEGRATION OF ML/DL WITH DEVOPS

### CHALLANGES:
**MLOps level 0 is common in many businesses that are beginning to apply ML to their use cases. This manual, data-scientist-driven process might be sufficient when models are rarely changed or trained. In practice, models often break when they are deployed in the real world. The models fail to adapt to changes in the dynamics of the environment, or changes in the data that describes the environment.**

### SOLUTION:
**To address the challenges of this manual process, MLOps practices for CI/CD and CT are helpful. By deploying an ML training pipeline, you can enable CT, and you can set up a CI/CD system to rapidly test, build, and deploy new implementations of the ML pipeline.**

## This project is a simple or basic version of the solution addressed above

# Task Description:
1. Create a container image that has Python3 and keras or numpy installed

2. When we launch this image, it should automatically start training the model in the container.
3. Create a job chain of job1 to job5 using build pipeline plugin in jenkins
4. Job-1: Pull the Github repository automatically when some developers push repository to Github.
5. Job-2: By looking at the code or program file, Jenkins should automatically start the respective machine learning
software installed interpreter, install image container to deploy code and start training.
6. Job-3: Train your model and predict accuracy or metrics
7. Job-4: If accuracy is less than 80%, then tweak the machine learning model architecture
8. Job-5: Retrain the model or notify that the best mode is being created
9. Create one extra job Job-6 to monitor: If the container where the app is running fails due to any reason then this job should automatically start the container again from the last trained model left

# My Steps involved in achieving the above tasks:
1.  I have used pytorch framework here instead of keras
I have used a prebuilt pytorch image availabel at [hub.docker.com](https://hub.docker.com)
Creating the docker file is very easy with this:
  1. I have created two dockerfiles:
     This one without hyperparameter tuning support
     ```
      FROM pytorch/pytorch

      CMD ["python", "train.py"]
     ```
     This one will create images with hyperparameter tuning support
     ```
      FROM pytorch/pytorch

      CMD ["python", "train.py", "-t True"]
     ```
     This accepts a command line arguments -t which is a type of boolean and if True it activates hyperparameter tuning that I have defined inside the training code.

2. This step will create a build pipeline of JOB-1 to JOB-5
   **Final look:**
   ![Image of Yaktocat](https://github.com/Wangsherpa/train_deep_learning_models/blob/master/images/build-pipeline.jpg)
   
3. JOB-1 : This will pull the github repo whenever it updates or changes using github webhook technique inside a folder `/pytorch`
        Choose this option in configuration and update the github webhook accordingly
       ![Github Poll SCM](https://github.com/Wangsherpa/train_deep_learning_models/blob/master/images/git-poll-scm.jpg)
       Command to copy all files pulled from github to the /pytorch folder
       ```
       sudo cp -v -r -f * /pytorch
       ```
4. JOB-2: This job will create a os image by looking at the code inside train.py 
          Here it will create a pytorch image as my code contains Convolutional Neural Network implemented using pytorch
          ![Pytorch](https://github.com/Wangsherpa/train_deep_learning_models/blob/master/images/job2-build.jpg)
          
**The above code in the image can be changed as below to meet the task requirements**
```
if cat /pytorch/network.py | grep Conv2d
then
  if sudo docker images | grep pytorch_train_without_hyper
  then
    echo "Required image already exist! Next job will run a container using this image"
  else
    echo "Creating the required Image..."

      if sudo docker build -t pytorch_train_without_hyper /pytorch-dockerfile/dockerfile1/
      then
        echo "Image created Successfully"
      else
        echo "Something went wrong while creating the image!"
      fi
  fi
else
  echo "Implement for other types of deeplearning and machine learning algorithms using else if statements"
```

5. JOB-3: This will create and run a container using an appropriate os image. Running this container will automatically start training the network for certain epochs and the test accuracy will be saved in the accuracy.txt file inside the same folder.
Adding the below command in the build->Execute shell
![Image](https://github.com/Wangsherpa/train_deep_learning_models/blob/master/images/job3.jpg)
**This code can also be changed as below to meet the task requirement**
```
if cat /pytorch/network.py | grep Conv2d
then
  sudo docker run -v /pytorch:/workspace pytorch_train_without_hyper
else
  echo "using else if statement and grep we can implement other functions in the similar way as this one"
```

**If everything goes well then the output will look like this:**
The accuracy is very low as I have trained this model for only 1 epoch.
![Image](https://github.com/Wangsherpa/train_deep_learning_models/blob/master/images/job3-output.jpg)

6. JOB-4: This job will fetch the accuracy saved in a file accuracy.txt and checks whether it meets the condition such as whether the accuracy is greater than 80% or not. If the accuracy is less than the expected one then this job will recreate an OS image using Docker file saved inside `/pytorch-dockerfiles/dockerfile2/Dockerfile`. The code for Dockerfile is already mentioned in step 1.
This way it checks for the condition and creates an image if it doesn't exists.
![Image](https://github.com/Wangsherpa/train_deep_learning_models/blob/master/images/job4-build.jpg)

**If everything goes well here then the output will look like this:**
![Output Image](https://github.com/Wangsherpa/train_deep_learning_models/blob/master/images/job4-output.jpg)

7. JOB-5: This job will train the network with hyperparameter tuning:
          Supported Hyperparameter to tune here is:
          Learning Rate and Optimizer
          However I could have added dropout, epochs , etc . But for now there is no any good resources about hyperparameter tuning in pytorch, so, I have implemented this using simple for loops and list of parameters to tune. My virtual box is not being able to use cuda so if I add many hyperparameters now this this will take very long time to train. That's why I have added only few hyperparameters to tune.
      This job will build only if the current accuracy is less than the required which is 80% in this case.
      ![Image](https://github.com/Wangsherpa/train_deep_learning_models/blob/master/images/job5-build.jpg)
      
      **Output if this JOB builds successfully** This is just a sample output
      Same will be send to the owner also, through email.
      ![Image](https://github.com/Wangsherpa/train_deep_learning_models/blob/master/images/job5-output.jpg)
      
      **Important Note about this job**
      After successfully completion of this job it will send email to the user/owner regardless of how much the accuracy is.
      It will send the best accuracy the model reached so far and best hyperparameters that were used.
      It's not true that if we tune the model if will perform well. The accuracy gets increased but it's not always possible that this hyperparameter pairs always perform well. So I have done this to avoid the model Jenkins job to enter infinite loop.
      **How will Jenkins Job enter infinite loop?**
      If the jobs are created in a way where the job2 will train a model and if the accuracy is not enough then the job4 will tune the hyperparameters and again invoke job2 to retrain. But in the worst case if none of the hyperparameter pairs give the required accuracy then this will keep executing like-> job2 will keep retraining and job4 will keep invoking job2 to retrain the model.
      
8. JOB-6 (Extra Job) : This job will run once in every week. This will send the post request to job2.
**Commands to send post request to job2 to train the model once in a week**
![Images](https://github.com/Wangsherpa/train_deep_learning_models/blob/master/images/job6-poll.jpg)
![Image](https://github.com/Wangsherpa/train_deep_learning_models/blob/master/images/job6-output.jpg)
**command** curl -X Post http://192.168.225.38:8080/view/Deep_learning/job/JOB-2/build?token=YourToken --user "username:password"

# Dataset Used:
# Convolutional Neural Networks
---
In this task, we train a **CNN** to classify images from the CIFAR-10 database.

The images in this database are small color images that fall into one of ten classes; some example images are pictured below.

![CIFAR](https://github.com/Wangsherpa/train_deep_learning_models/blob/master/images/cifar_data.png)

**Model Architecture Used**
![layers](https://github.com/Wangsherpa/train_deep_learning_models/blob/master/images/layers.png)
     
