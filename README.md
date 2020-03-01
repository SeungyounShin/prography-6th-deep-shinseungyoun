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
|\|version1|version2|
|------|---|---|
|accuracy|테스트2|테스트3|


## Conclusion

M(x) = F(x) + H(x)에서 H(x)의 커널사이즈를 7로 했을 때 accuracy가 떨어지는 현상이 발생함. 

즉, H(x)의 residual 정보는 backprogation 할 때 weight의 정보손실을 방지하기 위한 역할이 큼으로 H(x)를 deep 하고 complex 하게 가져갈 수록 훈련에서 불균형을 가져오게됨.

## Reference
Saining Xie Ross Girshick Piotr Dollar Zhuowen Tu1 Kaiming He. Aggregated Residual Transformations for Deep Neural Networks. In CVPR,2017.
