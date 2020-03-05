# prography-6th-deep-shinseungyoun
프로그라피 6기 과제

![model result](https://github.com/SeungyounShin/prography-6th-deep-shinseungyoun/blob/master/img/result.png?raw=true)

## Model
**version1**

skip connection 시 pooling으로 줄어든 feature와 indentity간의 dimension 보간을 위해 downsampling을 해줌.
최대한 simple한 H(x)를 설계하는 것이 목적이었기 때문에 H는 Conv1x1 ,strid=2 로 설정.

![Alt text](https://github.com/SeungyounShin/prography-6th-deep-shinseungyoun/blob/master/img/vgg16_skipconn_ver1.png?raw=true)

**version2**
[weight(https://drive.google.com/file/d/1QcuQFdd0oSvAjSO4ZiS7FZ187vY7tVWT/view?usp=sharing)

mnist데이터는 매우 작은편이기 때문에 굳이 큰 bottleneck 구조를 가져갈 필요가 없다고 판단.

Conv1를 bottleneck 으로 하면 channel 외에는 변화가 생기지 않도록 pooling을 제거
이후 identity에 Conv1x1으로 channel 보간 후 
M(x) = F(x) + H(x) 

![Alt text](https://github.com/SeungyounShin/prography-6th-deep-shinseungyoun/blob/master/img/vgg16_skipconn_ver2.png?raw=true)
## Accuracy

| table  | version1 | version2 |
| ------------- | ------------- |------------- |
| accuracy  | 88.38%  | 99.03%  |
| w/o skip-connection  | x  | 99.01%  |


## Conclusion

**version1** 구조에서 M(x) = F(x) + H(x);[1] H(x)의 커널사이즈를 7로 했을 때 accuracy가 떨어지는 현상이 발생하였다.

즉, H(x)의 residual 정보는 backprogation 할 때 weight의 정보손실을 방지하기 위한 역할이 큼으로 H(x)를 deep 하고 complex 하게 가져갈 수록 훈련에서 불균형을 가져오게된다는 가설을 세웠다.

가설을 확인하기 위해 H를 conv1x1, stride=2 로 설정했을 때 정확도가 상승하였다.

즉 H를 최대한 적게하고 Identity의 정보를 유지하는 방향으로 network 를 재설계하였다. 

conv1x1로 channel 수만 보간해주는 **version2** 와 같이 재설계를 진행하였는데 이렇게 하면 중간에 downsample을 하면안된다. mnist 데이터는 28x28로 사이즈가 매우 작기 때문에 conv1을 bottleneck으로 하는 네트워크를 생성하였다. 이렇게 skip connection을 구현하고 avgPool을 이용해 feature의 전반적인 정보를 좀 더 작은 dimension 으로 fusion 한 후 이를 fc 를 이용해 classification 과제를 수행하는 네트워크를 구성 정확도를 99% 까지 상승시켰다.

**version2** 의 경우 skip connection 이후의 activation map의 일부는 다음과 같다.
![Alt text](https://github.com/SeungyounShin/prography-6th-deep-shinseungyoun/blob/master/img/Figure_1.png?raw=true)

skip connection 후 feature map 이 geometric한 정보를 잘 나타낸 것을 토대로 앞에서 구현한 모델의 long path skip connection 구조가 feature 자체를 취득하는데는 큰 문제가 없는 것으로 보인다.

그러면 이 activation map 이 어떻게 생성되는지 시각화해보자.

![Alt text](https://github.com/SeungyounShin/prography-6th-deep-shinseungyoun/blob/master/img/change.png?raw=true)


앞의 skip connection 구조가 local evidence 를 더해주지만 실험적인 결과를 보면 큰 차이는 없는 듯하다. 아마 이러한 이유는 vgg16은 깊지 않기 때문에 degradation 이 크게 일어나지 않고 identity를 더해주지 않아도 크게 문제가 없기 때문이라고 생각된다. 하지만, accuracy 표에서 skip connection 을 이용한 구조가 미세한 정도로 정확도가 높은 것으로 보아 skip connection 이 기존 deep 한 구조의 훈련을 원활하게 하는 것보다 local evidence를 highlight 하는 효과를 조금더 적용하지 않나 생각된다.


## Reference
[0] Saining Xie Ross Girshick Piotr Dollar Zhuowen Tu1 Kaiming He. Aggregated Residual Transformations for Deep Neural Networks. In CVPR,2017.

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. In CVPR,2016.
