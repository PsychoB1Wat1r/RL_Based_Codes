import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import time


class CliffWalkingEnv:
    def __init__(self, n_col, n_row):
        self.n_row = n_row
        self.n_col = n_col
        self.x = 0  # 记录当前智能体位置的横坐标
        self.y = 0  # 记录当前智能体位置的纵坐标

    def step(self, action):  # 外部调用这个函数来改变当前位置
        # 4种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.n_col - 1, max(0, self.x + change[action][0]))
        self.y = min(self.n_row - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.n_col + self.x
        reward = -1
        done = False
        if self.y == self.n_row - 1 and self.x > 0:  # 下一个位置在悬崖或者目标
            done = True
            if self.x != self.n_col - 1:
                reward = -100
        return next_state, reward, done

    def reset(self):  # 回归初始状态,起点在左上角
        self.x = 0
        self.y = 0
        return self.y * self.n_col + self.x


class DynaQ:
    """ Dyna-Q算法 """

    def __init__(self, n_col, n_row, epsilon, alpha, gamma, n_planning, n_action=4):
        self.Q_table = np.zeros([n_row * n_col, n_action])  # 初始化Q(s,a)表格
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数
        self.n_planning = n_planning  # 执行Q-planning的次数, 对应1次Q-learning
        self.model = dict()  # 环境模型 根据字典的性质，若该数据本身存在于字典中，便不会再一次进行添加

    def take_action(self, state):  # 选取下一步的操作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def q_learning(self, s0, a0, r, s1):
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error  # alpha代表学习率 神经网络中就是梯度

    def update(self, s0, a0, r, s1):
        self.q_learning(s0, a0, r, s1)
        self.model[(s0, a0)] = r, s1  # 将数据添加到模型中
        for _ in range(self.n_planning):  # Q-planning循环
            # 随机选择曾经遇到过的状态动作对
            (s, a), (r, s_) = random.choice(list(self.model.items()))
            self.q_learning(s, a, r, s_)


def DynaQ_CliffWalking(n_planning):
    n_col = 12
    n_row = 4
    env = CliffWalkingEnv(n_col, n_row)
    epsilon = 0.01
    alpha = 0.1
    gamma = 0.9
    agent = DynaQ(n_col, n_row, epsilon, alpha, gamma, n_planning)
    num_episodes = 500  # 智能体在环境中运行多少条序列
    return_list = []  # 记录每一条序列的回报
    for i in range(10):  # 显示10个进度条
        # tqdm的进度条功能
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                    agent.update(state, action, reward, next_state)
                    state = next_state
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list


## 调用Q-planning算法
np.random.seed(0)
n_planning_list = [0, 2, 20]
for n_planning in n_planning_list:
    print('Q-planning步数为：%d' % n_planning)
    time.sleep(0.1)
    return_list = DynaQ_CliffWalking(n_planning)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list,
             return_list,
             label=str(n_planning) + ' planning steps')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Dyna-Q on {}'.format('Cliff Walking'))
plt.minorticks_on()
plt.grid(which='minor', linestyle='--', linewidth=0.75)
plt.show()