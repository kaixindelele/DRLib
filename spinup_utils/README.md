# 功能超全的强化学习画图脚本

仓库里附加了我刚刚全面测试的实验结果，下载本仓库后，直接执行下面的脚本，即可画图：

```
python spinup_utils/plot.py HER_DRLib_mpi1/2 --select Push
```

注意文件夹的路径要写对，其中HER_DRLib_mpi1为主文件夹名称，后面的2是子文件夹的第一个字母，一般来说这个必须得加，才能保证多个随机种子取平均，如果好奇的同学可以试试不同的路径会有什么效果~



相比于原始的Spinning up 的plot.py文件

## 原始画图效果：

![https://img-blog.csdnimg.cn/20210406162732714.png](https://img-blog.csdnimg.cn/20210406162732714.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hlaGVkYWRhcQ==,size_16,color_FFFFFF,t_70)

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
9.对图例legend根据性能从高到低排序，便于分析比较
10. 提供clip_xaxis值，对训练程度进行统一截断，图看起来更整洁。
11. **新功能**，在plot_demo_files里面添加了同一个实验、四组不同随机种子的数据，便于大家测试和调参。

## 使用步骤：
1. 按照初始界面下载安装本代码库；
2. 执行spinup_utils.plot.py
3. 出示例图：

例程图：

![https://img-blog.csdnimg.cn/20210512223911337.png](https://img-blog.csdnimg.cn/20210512223911337.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hlaGVkYWRhcQ==,size_16,color_FFFFFF,t_70)


注意：**seaborn版本0.8.1**


## 最终画图效果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021040616224411.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hlaGVkYWRhcQ==,size_16,color_FFFFFF,t_70)
如果大家还有什么新的功能，我也可以想办法加上去~

## 多个子图绘制和图例legend位置调整：
如下图，如果是多个子图的绘制，需要利用group_plot.py脚本：
[https://github.com/kaixindelele/DRLib/blob/main/spinup_utils/group_plot.py](https://github.com/kaixindelele/DRLib/blob/main/spinup_utils/group_plot.py)

这里的legend位置要调整到最底层，不能简单的用默认的best设置，需要替换成：


```python
plt.legend(bbox_to_anchor=(x, y))
```

这里的y为负数，代表在子图的下面~

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210611093742566.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hlaGVkYWRhcQ==,size_16,color_FFFFFF,t_70)


## 代码链接，有详细注释：
[https://github.com/kaixindelele/DRLib/blob/main/spinup_utils/plot.py](https://github.com/kaixindelele/DRLib/blob/main/spinup_utils/plot.py)

## matplotlib均值和方差图-多组成功率为例-代码

![多组实际效果](https://img-blog.csdnimg.cn/20210221004735541.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hlaGVkYWRhcQ==,size_16,color_FFFFFF,t_70#pic_center)

[多组成功率代码](https://github.com/kaixindelele/DRLib/blob/main/spinup_utils/plot_success_group.py)

![单组成功率柱状图](https://img-blog.csdnimg.cn/20210408120845755.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hlaGVkYWRhcQ==,size_16,color_FFFFFF,t_70)

[单组成功率代码](https://github.com/kaixindelele/DRLib/blob/main/spinup_utils/plot_success.py)

## 关于强化学习绘图的其他优质教程：
尤其是关于tsplot和lineplot函数的使用：
以及设置画一些和spinningup格式不一样的图，可以参考下面的教程和代码。
但是如果画多组实验，全部功能的图，最好还是用我那个~


1. [启人大佬：强化学习实验中的绘图技巧-使用seaborn绘制paper中的图片](https://zhuanlan.zhihu.com/p/75477750)
2. [使用seaborn绘制强化学习中的图片](https://zhuanlan.zhihu.com/p/147847062)
3. [强化学习横轴纵轴含义和画图基准细节](参考链接：https://spinningup.openai.com/en/latest/spinningup/bench.html#experiment-details)
4. rl_plotter,可以直接下载使用





## 联系方式：
ps: 欢迎做强化的同学加群一起学习：

深度强化学习-DRL：799378128

欢迎关注知乎帐号：[未入门的炼丹学徒](https://www.zhihu.com/people/heda-he-28)

CSDN帐号：[https://blog.csdn.net/hehedadaq](https://blog.csdn.net/hehedadaq)

极简spinup+HER+PER代码实现：[https://github.com/kaixindelele/DRLib](https://github.com/kaixindelele/DRLib)
