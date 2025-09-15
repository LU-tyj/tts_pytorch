# Tacotron&Tacotron2

参考：[知乎](https://zhuanlan.zhihu.com/p/101064153?utm_source=wechat_session&utm_medium=social&utm_oi=651550372523675648&accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NDk5MDI3MzgsImZpbGVHVUlEIjoiOGhHM0RWVEdXdFJSdmt2OCIsImlhdCI6MTc0OTkwMjQzOCwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOi05MTQ4NzU1MTc5fQ.xTzK549qxnm3tjvtd76h-XSuAtcjPloSg3omVhoYKJ4)，[一些名词](https://zhuanlan.zhihu.com/p/99122527)

这是上篇文章中提到的方法3，基于深度学习的端到端语音合成模型它的优点就是可以仅通过输入(text, wav)的pair来进行学系，也不用手动提取特征，也不需要细致组合各种模块。本文将介绍**Tacotron**及其改良版**Tacotron2**

---

## 1. 大前辈 [Wavenet](https://en.wikipedia.org/wiki/WaveNet)

 Wavenet并不是一个端到端模型，由于它的输入并不是raw text而是经过处理的特征，因此它实际上只是代替了传统TTS pipeline的后端

其最大的成功之处是使用dilated causal convolution技术来增加CNN的receptive field，从而提升了模型建模long dependency的能力，如下图所示![dilated casual convoluntion](https://upload.wikimedia.org/wikipedia/en/8/8b/WaveNet_animation.gif)

其还是用了残差链接、门控机制等，在tts中取得了较好的表现

---

## 2. Tacotron

Tacotron是第一个端对端的TTS神经网络模型，输入**raw text**，Tacotron可以直接输出**mel-spectrogram**，再利用**Griffin-Lim**算法就可以生成波形了。模型的总体架构如下图所示：![Tacotron](https://pic2.zhimg.com/v2-2f24ef52b9698852c415aea074f901d5_1440w.jpg)

总的来说和s2s很像，大体由encoder和decoder组成

下面具体介绍这些模块

### 2.1 CBHG

用来从序列中提取高层次特征的模块

![CBHG](https://picx.zhimg.com/v2-5b826f2065171228fa1e0cded2e43d33_1440w.jpg)

输入一个序列输出的也是序列

### 2.2 Encoder

输入被CBHG处理之前还需要经过pre-net进行预处理，作者设计pre-net的意图是让它成为一个bottleneck layer来提升模型的泛化能力，以及加快收敛速度。

pre-net是由全连接层和dropout层组成

![NN](https://pic2.zhimg.com/v2-7530fc1123838af2772b5f9f0baed88f_1440w.jpg)

### 2.3 Decoder

作者使用两个decoder: attention decoder和output decoder，attention decoder用来生成query vector作为attention的输入,attention生成context vector，最后output decoder则将query vector和context vector组合在一起作为输入。

这里没有直接用output decoder生成spectrogram，而是生成了80-bend mel-scale spectrogram。由于spectrogram的size太大了，这样操作虽然损失了信息，但是打打节省了时间。

另外一个用来缩减计算量的做法是每个decoder step预测多个(r个)frame，且作者发现这样做还可以加速模型的收敛。

### 2.4 post-processing net and waveform synthesis

作者使用比较简单的Griffin-Lim 算法来生成最终的波形，由于decoder生成的是mel-spectrogram，因此需要转换成linear-scale spectrogram才能使用Griffin-Lim算法，这里作者同样使用CBHG来完成这个任务。

实际上这里post-processing net中的CBHG是可以被替换成其它模块用来生成其它东西的，比如直接生成waveform，在Tacotron2中，CBHG就被替换为Wavenet来直接生成波形。

---

## 3. Tacotron2

Tacotron最后生成最终波形的算法太简单了，因此需要找一个强大的vocoder来改进

### 3.1 Model

Tacotron2使用了一个和Wavenet十分相似的模型来代替Griffin-Lim算法，同时也对Tacotron模型的一些细节也做了更改，最终生成了十分接近人类声音的波形。

模型的架构如下图所示：

![](https://pic3.zhimg.com/v2-0834e55127954fa0233b1abcdbec443c_1440w.jpg)

与Tacotron的不同之处在于：

- 不使用CBHG，而是使用普通的LSTM和Convolution layer
- decoder每一步只生成一个frame
- 增加post-net，即一个5层CNN来精调mel-spectrogram

---

## 4. 总结

Tacotron&Tacotron2通过生成mel- spectrogram来减少训练所需的时间，Tacotron2最主要的进步是将mel- spectrogram转化成waveform这一步从使用普通的Griffin-lim算法改为使用类似Wavenet的模型，从而提高了效果