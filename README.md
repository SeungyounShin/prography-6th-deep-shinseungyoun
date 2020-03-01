# prography-6th-deep-shinseungyoun
프로그라피 6기 과제

![model result](https://github.com/SeungyounShin/prography-6th-deep-shinseungyoun/blob/master/img/result.png?raw=true)

## Model
**version1**

skip connection 시 pooling으로 줄어든 feature와 indentity간의 dimension 보간을 위해 downsampling을 해줌.
최대한 simple한 H(x)를 설계하는 것이 목적이었기 때문에 H는 Conv1x1 ,strid=2 로 설정.

![Alt text](https://github.com/SeungyounShin/prography-6th-deep-shinseungyoun/blob/master/img/vgg16_skipconn_ver1.png?raw=true)

**version2**

mnist데이터는 매우 작은편이기 때문에 굳이 큰 bottleneck 구조를 가져갈 필요가 없다고 판단.

Conv1를 bottleneck 으로 하면 channel 외에는 변화가 생기지 않도록 pooling을 제거
이후 identity에 Conv1x1으로 channel 보간 후 
M(x) = F(x) + H(x) 

![Alt text](https://github.com/SeungyounShin/prography-6th-deep-shinseungyoun/blob/master/img/vgg16_skipconn_ver2.png?raw=true)
## Accuracy

| table  | version1 | version2 |
| ------------- | ------------- |------------- |
| accuracy  | 88.38%  | 99.03%  |


## Conclusion

**version1** 구조에서 M(x) = F(x) + H(x); H(x)의 커널사이즈를 7로 했을 때 accuracy가 떨어지는 현상이 발생함. 

즉, H(x)의 residual 정보는 backprogation 할 때 weight의 정보손실을 방지하기 위한 역할이 큼으로 H(x)를 deep 하고 complex 하게 가져갈 수록 훈련에서 불균형을 가져오게된다는 가설을 세움.

가설을 확인하기 위해 H를 conv1x1, stride=2 로 설정했을 때 정확도가 상승함.

즉 H를 최대한 적게하고 Identity의 정보를 유지하는 방향으로 network 를 재설계함. 

conv1x1로 channel 수만 보간해주는 **version2** 와 같이 재설계를 진행하였는데 이렇게 하면 중간에 downsample을 하면안됨. mnist 데이터는 28x28로 사이즈가 매우 작기 때문에 conv1을 bottleneck으로 하는 네트워크를 생성함. 이렇게 skip connection을 구현하고 avgPool을 이용해 feature의 전반적인 정보를 좀 더 작은 dimension 으로 fusion 한 후 이를 fc 를 이용해 classification 과제를 수행하는 네트워크를 구성 정확도를 99% 까지 상승시킴.

**version2** 의 경우 skip connection 이후의 activation map의 일부는 다음과 같다.
![Alt text](https://github.com/SeungyounShin/prography-6th-deep-shinseungyoun/blob/master/img/Figure_1.png?raw=true)

그러면 이 activation map 이 어떻게 생성되는지 시각화해보자.

![Alt text](https://github.com/SeungyounShin/prography-6th-deep-shinseungyoun/blob/master/img/change.png?raw=true)


## Reference
Saining Xie Ross Girshick Piotr Dollar Zhuowen Tu1 Kaiming He. Aggregated Residual Transformations for Deep Neural Networks. In CVPR,2017.
