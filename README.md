# AutoEncoder_Classifier
## Dependencies
- Driver Version: 430.26
- CUDA Version: 10.2
- Docker version: 19.03.2

## Steps
1. Install
```sh
$ git clone git@github.com:ryuji0123/cv_dnn.git
```

2. Environment Setup

The names of the docker image and container are specified by constants described in docker/env.sh.
These constants can be edited to suit your project.
```sh
$ cd cv_dnn
$ sh docker/build.sh
$ sh docker/run.sh
$ sh docker/exec.sh
```

4. Run Training Steps
```sh
$ sh nohup_train.sh
```
If you want to stop seeing stdout without breaking train steps, you can just press ctrl+c. The above shell uses nohup and run python script in background, you don't pay attention to how to avoid intrupting train steps. See Logger / Trainer module in detail.

5. See Results
```sh
$ sh app_ml.sh
```
If you want to see train steps and its status with both stdout and mlflow, you can use tmux. We install it in docker image.
After training process started, it can be seen on mlflow's application. You can use existing shell script:

```sh
$ sh app_ml.sh
```

Then, please check  localhost:7979 on your web browser.

### Docker
There are three steps to create your own docker environment.
- Build docker image. You can use existing shell script:
```sh
$ sh docker/build.sh
```
Since default user in docker container is root user, user_id and group_id in docker are different from them in host OS. This causes permission problems if you generate files in docker container. To fix this problem, we create duser (docker user). It has the same user_id and group_id with them in host OS, so if you write or edit files, you can access the same files in host OS. Also, duer can use sodo command in docker container without a password, so you don't have to pay attention to settings when you want to install libraries.

- Run docker container. You can use existing shell script:
```sh
$ sh docker/run.sh
```

- Exec docker container. You can use existing shell script:
```sh
$ sh docker/exec.sh
```

### Main
Here's where you combine all previous part.

1. Parse console arguments and obtain configurations as "args".
1. Run "main" function.
1. Create a mlflow session.
1. Create an instance of "Model", "Dataset", "DataLoader"
1. Create an instance of "Trainer" and pass "Model" and "DataLoader" to it.
1. Now you can train your model by calling "trainer.fit()"

## Acknowledgement
This repository is forked from [ryuji0123](https://github.com/ryuji0123/cv_dnn)
