# tabular_deep_cfr_kuhn.py
#
# 一个极简版的 Tabular CFR（可以当成 Deep CFR 的无网络 baseline）
# 只跑 kuhn_poker，结果是：NashConv 应该收敛到 ~0

import numpy as np
import pyspiel
from open_spiel.python.algorithms import exploitability


class TabularCFRKuhn:
    def __init__(self, game):
        self._game = game
        self._num_players = game.num_players()

        # regret_sum[player][info_state_str] -> np.array(num_actions)
        self._regret_sum = [{} for _ in range(self._num_players)]
        # strategy_sum[player][info_state_str] -> np.array(num_actions)
        self._strategy_sum = [{} for _ in range(self._num_players)]

    # --- 工具函数：基于 regret 做 regret-matching 得到当前策略 ---
    def _get_strategy(self, player, info_state_str, legal_actions):
        num_actions = len(legal_actions)
        regrets = self._regret_sum[player].get(
            info_state_str, np.zeros(num_actions, dtype=np.float32)
        )

        # regret matching: 正 regret 归一化，否则用 uniform
        positive_regrets = np.maximum(regrets, 0.0)
        sum_positive = positive_regrets.sum()
        if sum_positive > 1e-12:
            strategy = positive_regrets / sum_positive
        else:
            strategy = np.ones(num_actions, dtype=np.float32) / num_actions
        return strategy

    # --- 更新 strategy_sum，用 reach-prob 加权 ---
    def _accumulate_strategy(self, player, info_state_str, strategy, reach_prob):
        if info_state_str not in self._strategy_sum[player]:
            self._strategy_sum[player][info_state_str] = np.zeros_like(strategy)
        self._strategy_sum[player][info_state_str] += reach_prob * strategy

    # --- 标准 CFR recursion ---
    def _cfr(self, state, player, reach_probs):
        """返回从当前 state 开始，对 player 的期望收益。"""

        if state.is_terminal():
            returns = state.returns()
            return returns[player]

        if state.is_chance_node():
            # Kuhn 这里 chance 就是发牌
            outcomes = state.chance_outcomes()
            value = 0.0
            for action, prob in outcomes:
                next_state = state.child(action)
                value += prob * self._cfr(next_state, player, reach_probs)
            return value

        current_player = state.current_player()
        info_state_str = state.information_state_string(current_player)
        legal_actions = state.legal_actions()
        num_actions = len(legal_actions)

        strategy = self._get_strategy(current_player, info_state_str, legal_actions)

        # 计算每个 action 的 utility
        util = np.zeros(num_actions, dtype=np.float32)
        node_util = 0.0

        for i, a in enumerate(legal_actions):
            next_state = state.child(a)
            next_reach_probs = list(reach_probs)

            # 更新 reach prob
            next_reach_probs[current_player] *= strategy[i]

            util[i] = self._cfr(next_state, player, next_reach_probs)
            node_util += strategy[i] * util[i]

        # 如果是我们更新的那个 player，就更新 regret_sum
        if current_player == player:
            # 对 CFR，regret 的权重是对手的 reach probability
            opponent = 1 - player
            regret_weight = reach_probs[opponent]
            regrets = util - node_util

            if info_state_str not in self._regret_sum[player]:
                self._regret_sum[player][info_state_str] = np.zeros_like(regrets)
            self._regret_sum[player][info_state_str] += regret_weight * regrets

        # 不管是不是我们自己，都更新一下 strategy_sum（用于平均策略）
        reach_prob_current = reach_probs[current_player]
        self._accumulate_strategy(current_player, info_state_str, strategy, reach_prob_current)

        return node_util

    def iterate(self, num_iters: int):
        """跑 num_iters 轮 CFR。"""
        for it in range(1, num_iters + 1):
            state = self._game.new_initial_state()
            # 对两个玩家分别跑一遍 CFR（这是 external CFR 常见写法）
            for p in range(self._num_players):
                self._cfr(state, player=p, reach_probs=[1.0, 1.0])

            if it % 1000 == 0:
                avg_policy = self.average_policy()
                conv = exploitability.nash_conv(self._game, avg_policy)
                print(f"[Iter {it}] NashConv = {conv:.6f}")

    # --- 把 tabular 平均策略封装成 pyspiel.Policy 兼容的对象 ---
    def average_policy(self):
        """返回一个 pyspiel.Policy 风格的对象，用于计算 NashConv。"""

        game = self._game
        strategy_sum = self._strategy_sum

        class _AvgTabularPolicy(pyspiel.Policy):
            def __init__(self, game, strategy_sum):
                super().__init__(game, list(range(game.num_players())))
                self._strategy_sum = strategy_sum

            def action_probabilities(self, state, player_id=None):
                if player_id is None:
                    player_id = state.current_player()

                info_state_str = state.information_state_string(player_id)
                legal_actions = state.legal_actions()

                if info_state_str not in self._strategy_sum[player_id]:
                    # 如果没见过这个信息状态，就用均匀策略
                    num_actions = len(legal_actions)
                    return {
                        a: 1.0 / num_actions for a in legal_actions
                    }

                s_sum = self._strategy_sum[player_id][info_state_str].copy()
                # 只保留合法动作
                probs = {}
                total = 0.0
                for i, a in enumerate(legal_actions):
                    total += s_sum[i]
                if total <= 1e-12:
                    num_actions = len(legal_actions)
                    for a in legal_actions:
                        probs[a] = 1.0 / num_actions
                else:
                    for i, a in enumerate(legal_actions):
                        probs[a] = s_sum[i] / total
                return probs

        return _AvgTabularPolicy(game, strategy_sum)


def main():
    game = pyspiel.load_game("kuhn_poker")
    solver = TabularCFRKuhn(game)

    print("Running tabular CFR on kuhn_poker for 20000 iterations...")
    solver.iterate(20000)

    avg_policy = solver.average_policy()
    conv = exploitability.nash_conv(game, avg_policy)
    print(f"Final NashConv: {conv:.6f}")


if __name__ == "__main__":
    main()