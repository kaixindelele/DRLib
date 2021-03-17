关于重新采集目标的活儿，如果通过debug的方式，直接打印每次拿到的值，可以实现，但是比较复杂；
如果看论文中的伪代码，那就太简单了，其中的变化，以及可能存在的隐患，几行伪代码是看不出来的。
因此我们做一个简单的推演：
先列出源代码，去除对数据的处理，只保留和her相关的语句：

```python
"""
    假设episode_trans里面有50个steps；
    从step1开始，step50结束；
    我们只考虑其中的step1，step7,8,9，以及step49；
    整个回合的运动状态可以用下面的流程简化：
    ----------------------------------------------------------------------------------
    step1-->   ...   -->step7-->       -->step8            -->step9          -->step49
      achieved_goal1 == achieved_goal7 != achieved_goal8  !=  achieved_goal9 == achieved_goal49
      desired_goal   == desired_goal   == desired_goal    ==  desired_goal   == desired_goal
      a1                a7                a8                  a9                a49
               没动               动了                 动了             没动
    ---------------------------------------------------------------------------------
    1.在step1到step7中，achieved_goal一直不等于desired_goal,
    2.不用HER,由于step_next(achieved_goal)!=step(desired_goal),奖励reward都是-1；
    3.直接稀疏奖励是学不到什么的；
    4.
    5.在用了HER的时候，如果achieved_goal一直不变，即智能体没有动作使得goal变化；
    6.那么HER修改目标的时候使得step(desired_goal)==step_next(achieved_goal)，因此奖励每次都是0；
    7.这时候会让智能体有一种躺平的错觉，我什么都不用动，都不用受到惩罚，那还努力个什么劲儿？
    8.
    9.好在我们存留了1/5的原始数据，即当什么都不动的时候，仍然会有-1的惩罚；
    10.这属于极大的隐患，如果我们1/5的数据不能约束智能体，朝着更好的方向；
    11.或者是随机探索不能让智能体接触目标，那么他永远都学不会好的策略！
    12.
    13.不考虑这么极端的情况，例子中我们在step7,8的时候都使得goal发生了变化；
    14.对于step7，在用HER采样future目标的时候，可以在step8-50，采样四个achieved_goal当做目标；
      13.当采取了step8时，new_desired_goal=achieved_goal8, 拿到的奖励是0
        14.直接完成任务，下次遇到s=(achieved_goal7,desired_goal=achieved_goal8)时，动作采取a7就完事儿了；
      15.当采取了step9时，new_desired_goal=achieved_goal9, 拿到的奖励是-1
        16.但是往后迭代，只考虑轨迹step7-step8-step9这个轨迹：
        17.Q((ag7,ag9),a7)=-1+gamma*Q((ag8,ag9),a8)，如果用了HER, 这个Q((ag8,ag9),a8)=0
        18.那么Q((ag7,ag9),a7)的值也会相应变高；
        19.上面的分析，可能有点迷惑，我写的也不严谨。
        20.大概的意思是，如果采样出的目标，不是能一步到达的，给的奖励也是-1，但对最终的学习也有帮助；
        21.如果大家能捋出更符合MDP的逻辑来就更好了；

    假设在step7的时候，采样了step9的achieved_goal9当desired_goal，那么可以将流程图简化成如下：        
    ----------------------------------------------------------------------------------
    -->step7-->             -->step8            -->step9          -->step49
       achieved_goal7       != achieved_goal8  !=  achieved_goal9 == achieved_goal49
       desired_goal         == desired_goal   == desired_goal    ==  desired_goal   == desired_goal
       a7      ↓                 a8                  a9                a49
               ↓     动了                 动了             没动
       achieved_goal9
       new_reward ↓
       int(achieved_goal9==achieved_goal8)
    ---------------------------------------------------------------------------------
    
"""
for transition_idx, transition in enumerate(episode_trans):
    # 先存一组原始数据，奖励一定也是稀疏的！dense的就不好使了~
    obs, action, reward, next_obs, done, info = copy.deepcopy(transition)
    store_transition(transition=(obs_arr, action, reward, next_obs_arr, done))
    
    sampled_goals = sample_achieved_goals(episode_trans, transition_idx,
                                          n_sampled_goal=self.n_sampled_goal)
    for goal in sampled_goals:
        obs, action, reward, next_obs, done, info = copy.deepcopy(transition)
        obs['desired_goal'] = goal
        next_obs['desired_goal'] = goal
        # Update the reward according to the new desired goal
        reward = reward_func(next_obs['achieved_goal'],
                             goal, info)
        store_transition(transition=(obs_arr, action, reward, next_obs_arr, done))
```
