import copy


# 经过 5 次策略评估和策略提升的循环迭代，策略收敛了。可以看到，解决同样的训练任务，价值迭
# 代总共进行了数十轮，而策略迭代中的策略评估总共进行了数百轮，价值迭代中的循环次数远少于策略迭代。

class CliffWalkingEnv:
    """ 悬崖漫步环境"""

    def __init__(self, n_col=12, n_row=4):
        self.n_col = n_col  # 定义网格世界的列
        self.n_row = n_row  # 定义网格世界的行
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.createP()

    def createP(self):
        # 初始化
        P = [[[] for j in range(4)] for i in range(self.n_row * self.n_col)]
        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.n_row):
            for j in range(self.n_col):
                for a in range(4):
                    # 如果位置已经在悬崖或者目标状态（终点）,此时智能体因无法继续移动，即无法与环境交互,
                    # 则任何动作奖励都为0,类似于网格边界（悬崖）初始化
                    if i == self.n_row - 1 and j > 0:
                        P[i * self.n_col + j][a] = [(1, i * self.n_col + j, 0, True)]
                        continue
                    # 非边界区域,即其他位置 这两行代码用于计算在执行动作a后的下一个状态的横纵坐标 max和min是为了防止不超过网格边界
                    next_x = min(self.n_col - 1, max(0, j + change[a][0]))
                    next_y = min(self.n_row - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.n_col + next_x
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.n_row - 1 and next_x > 0:
                        done = True
                        if next_x != self.n_col - 1:  # 不在右下角的重点 那么位置就到达了悬崖
                            reward = -100
                    P[i * self.n_col + j][a] = [(1, next_state, reward, done)]
        return P


class PolicyIteration:
    """ 策略迭代算法 """
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.n_col * self.env.n_row  # 初始化价值为0
        self.pi = [[0.25, 0.25, 0.25, 0.25]
                   for i in range(self.env.n_col * self.env.n_row)]  # 初始化为均匀随机策略
        self.theta = theta  # 策略评估收敛阈值
        self.gamma = gamma  # 折扣因子

    def policy_evaluation(self):  # 策略评估
        cnt = 1  # 计数器
        while 1:
            max_diff = 0
            new_v = [0] * self.env.n_col * self.env.n_row
            for s in range(self.env.n_col * self.env.n_row):
                qsa_list = []  # 开始计算状态s下的所有Q(s,a)价值
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                        # 本章环境比较特殊,奖励和下一个状态有关,所以需要和状态转移概率相乘
                    qsa_list.append(self.pi[s][a] * qsa)
                new_v[s] = sum(qsa_list)  # 状态价值函数和动作价值函数之间的关系
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break  # 满足收敛条件,退出评估迭代
            cnt += 1
        print("策略评估进行%d轮后完成" % cnt)

    def policy_improvement(self):  # 策略提升
        for s in range(self.env.n_row * self.env.n_col):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            max_q = max(qsa_list)  # 找到动作价值函数的最大值max_Q
            cnt_q = qsa_list.count(max_q)  # 可能存在执行不同动作a后均可达到最大的Q值 因此需要计算有几个动作得到了最大的Q值
            self.pi[s] = [1 / cnt_q if q == max_q else 0 for q in qsa_list]  # 让这些动作均分概率
        print("策略提升完成")
        return self.pi

    def policy_iteration(self):  # 策略迭代
        while 1:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)  # 将列表进行深拷贝,方便接下来进行比较
            new_pi = self.policy_improvement()
            if old_pi == new_pi:
                break

def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值：")
    for i in range(agent.env.n_row):
        for j in range(agent.env.n_col):
            # 为了输出美观,保持输出6个字符
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.n_col + j]), end=' ')
        print()

    print("策略：")
    for i in range(agent.env.n_row):
        for j in range(agent.env.n_col):
            # 一些特殊的状态,例如悬崖漫步中的悬崖 坠入悬崖
            if (i * agent.env.n_col + j) in disaster:
                print('-_-', end=' ')
            elif (i * agent.env.n_col + j) in end:  # 目标状态 到达终点
                print('^_^', end=' ')
            else:
                a = agent.pi[i * agent.env.n_col + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'x'
                print(pi_str, end=' ')
        print()


class ValueIteration:
    """ 价值迭代算法 """
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.n_col * self.env.n_row  # 初始化价值为0
        self.theta = theta  # 价值收敛阈值
        self.gamma = gamma
        # 价值迭代结束后得到的策略
        self.pi = [None for i in range(self.env.n_col * self.env.n_row)]

    def value_iteration(self):
        cnt = 0
        while 1:
            max_diff = 0
            new_v = [0] * self.env.n_col * self.env.n_row
            for s in range(self.env.n_col * self.env.n_row):
                qsa_list = []  # 开始计算状态s下的所有Q(s,a)价值
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                    qsa_list.append(qsa)  # 这一行和下一行代码是价值迭代和策略迭代的主要区别
                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta: break  # 满足收敛条件,退出评估迭代
            cnt += 1
        print("价值迭代一共进行%d轮" % cnt)
        self.get_policy()

    def get_policy(self):  # 根据价值函数导出一个贪婪策略
        for s in range(self.env.n_row * self.env.n_col):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            max_q = max(qsa_list)
            cnt_q = qsa_list.count(max_q)  # 计算有几个动作得到了最大的Q值
            # 让这些动作均分概率
            self.pi[s] = [1 / cnt_q if q == max_q else 0 for q in qsa_list]


## 调用策略迭代
env = CliffWalkingEnv()
action_meaning = ['👆', '👇', '👈', '👉']
theta = 0.001
gamma = 0.9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47])
## 调用价值迭代
print('*********************************************************************\n'
      '*********************************************************************')
agent = ValueIteration(env, theta, gamma)
agent.value_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47])
print(env.P)