# Deep Learning Project Group Repository

## 目录
等你补充

## 项目架构
等你补充，可以用GPT image 2.0画架构图

## MAC推送需要注意的点
① 配置好.gitignore
② 脚本权限打开：
git config --global pull.rebase false  
sudo chmod -R 777 <相对路径>

# 这个可以保留我觉得，因为我自己要看，放在最后就好
# 更新日志From Qiu ZS
## 5.5-Update in GitHub
- 1. 上传了官方baseline，已训练未调参版本的。  
- 2. 自己写的InterFormer，两个版本。V0.1是没有调参的，还在训练中。V0.2是微调后的，在排队，预计明天中午可以练好，也许吧。后续主要围绕InterFormer架构优化。  
- 3. 已优化方面：硬件设施适配，Loss function等  
- 4. 未优化方面：还是要解决一下速率问题。另外调参还要注意学习率，补充一下梯度。明天分析一下这三个训练的结果。最后还有特征工程没做。

## 5.7-Update in GitHub
- 1. 上传了InterFormer初版和V0.1与V0.3版本相关文件与训练结果，详情见Readme
- 2. 上传了最新InterFormerV0.4版本，预计中午训练结束
- 3. 下一步重点放在特征工程与调参方面

## 5.8 -Update in GitHub
- 1. 上传了Hyformer 0.2的训练与评估结果
- 2. 上传了Interformer v0.5的模型，正在训练，预计明天中午训练结束
- 3. 官方的baseline真牛逼，还是要敬畏

## 5.8 -Update in GitHub
- 1. 上传了v0.5的训练结果并做简单分析，infer超时了唉
- 2. 上传了全量数据的schema.json并做简单分析，为下一步优化提供方向
- 3. 接下来同步优化interformer和Hyformer，看哪个表现好

## 5.8 -Update in GitHub
- 1. 消融实验 trainning中 （问题：GPU memory没有充分利用，硬件优化需要提升）
- 2. 对数item频次 增加dense特征  提升了一点点
- 3. 贝叶斯平滑CTR 提升较大！

## 5.8 -Update in GitHub
- 1. 结合对数item频次和贝叶斯平滑CTR的基础上做embedding（已写好，排队中）
