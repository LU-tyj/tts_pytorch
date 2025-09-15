# pytorch 复现 tacotron1

## 参考文献：

- [知乎tacotron实验](https://zhuanlan.zhihu.com/p/114212581?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NTY5ODk4NjEsImZpbGVHVUlEIjoiOGhHM0RWVEdXdFJSdmt2OCIsImlhdCI6MTc1Njk4OTU2MSwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOi05MjM1ODYwMjQ2fQ.puZj_NeiUT9_g9K3-wJCRUeb-fyntVBNI4J8-Mk8kG4)

- [keithito/tacotron](https://github.com/keithito/tacotron)
- [Kyubyong/tacotron](https://github.com/Kyubyong/tacotron)
- [r9y9/tacotron](https://github.com/r9y9/tacotron_pytorch)

- [知乎tacotron基本知识](https://zhuanlan.zhihu.com/p/101064153?utm_source=wechat_session&utm_medium=social&utm_oi=651550372523675648&accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NDk5MDI3MzgsImZpbGVHVUlEIjoiOGhHM0RWVEdXdFJSdmt2OCIsImlhdCI6MTc0OTkwMjQzOCwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOi05MTQ4NzU1MTc5fQ.xTzK549qxnm3tjvtd76h-XSuAtcjPloSg3omVhoYKJ4)
f

---

## 基本知识

[tacotron basic](./basic.md)

内容源自[知乎tacotron基本知识](https://zhuanlan.zhihu.com/p/101064153?utm_source=wechat_session&utm_medium=social&utm_oi=651550372523675648&accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NDk5MDI3MzgsImZpbGVHVUlEIjoiOGhHM0RWVEdXdFJSdmt2OCIsImlhdCI6MTc0OTkwMjQzOCwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOi05MTQ4NzU1MTc5fQ.xTzK549qxnm3tjvtd76h-XSuAtcjPloSg3omVhoYKJ4)，总结了相关的tts模型发展的历史，以及相关的大致的模型结构。


---

## 数据来源 

https://www.openslr.org/18/

数据目录结构：
```bash
database
├── data 有全部的数据文件wav和trn
├── dev
├── lm_phone
├── lm_word
├── README.TXT 具体内容
├── test
└── train 训练部分，其中trn中的是内容为地址信息
```

---

## 目录

```bash
tts
├── basic.md
├── data.py 原数据处理
├── database 原数据文件夹
├── module 
│   ├── attention.py
│   └── tacotron.py
├── README.md
├── run.py 运行代码
└── train.py 训练函数
```

## 运行

```bash
export KMP_DUPLICATE_LIB_OK=TRUE # 遇到了存在多个Openmp库的报错
python run.py
```


会生成output文件夹，其中有train和test，分别放两者的mel spectroram