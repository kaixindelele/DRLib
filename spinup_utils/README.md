# 功能超全的强化学习画图脚本

相比于原始的Spinning up 的plot.py文件

## 原始画图效果：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210406162732714.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hlaGVkYWRhcQ==,size_16,color_FFFFFF,t_70)

线条多一点，就根本分不清谁是谁了。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210406162852952.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hlaGVkYWRhcQ==,size_16,color_FFFFFF,t_70)



## 加了九大功能：

1. 可以直接在pycharm或者vscode执行，也可以用命令行传参；
2. 按exp_name排序，而不是按时间排序；
3. 固定好每个exp_name的颜色；
4. 可以调节曲线的线宽，便于观察；
5. 保存图片到本地，便于远程ssh画图~
6. 自动显示全屏
7. 图片自适应
8. 针对颜色不敏感的人群,可以在每条legend上注明性能值,和性能序号
9. **新功能**，在plot_demo_files里面添加了同一个实验、四组不同随机种子的数据，便于大家测试和调参。

例程图：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210512223911337.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hlaGVkYWRhcQ==,size_16,color_FFFFFF,t_70)


注意：**seaborn版本0.8.1**


## 最终画图效果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021040616224411.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hlaGVkYWRhcQ==,size_16,color_FFFFFF,t_70)
如果大家还有什么新的功能，我也可以想办法加上去~

## 代码链接，有详细注释：
[https://github.com/kaixindelele/DRLib/blob/main/spinup_utils/plot.py](https://github.com/kaixindelele/DRLib/blob/main/spinup_utils/plot.py)

## 关于强化学习绘图的其他优质教程：
尤其是关于tsplot和lineplot函数的使用：
以及设置画一些和spinningup格式不一样的图，可以参考下面的教程和代码。
但是如果画多组实验，全部功能的图，最好还是用我那个~


1. [启人大佬：强化学习实验中的绘图技巧-使用seaborn绘制paper中的图片](https://zhuanlan.zhihu.com/p/75477750)
2. [使用seaborn绘制强化学习中的图片](https://zhuanlan.zhihu.com/p/147847062)





## 联系方式：
ps: 欢迎做强化的同学加群一起学习：

深度强化学习-DRL：799378128

欢迎关注知乎帐号：[未入门的炼丹学徒](https://www.zhihu.com/people/heda-he-28)

CSDN帐号：[https://blog.csdn.net/hehedadaq](https://blog.csdn.net/hehedadaq)

极简spinup+HER+PER代码实现：[https://github.com/kaixindelele/DRLib](https://github.com/kaixindelele/DRLib)
