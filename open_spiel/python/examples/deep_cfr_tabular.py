# tabular_deep_cfr_kuhn.py
#
# A small "tabular Deep CFR" / CFR implementation for Kuhn poker.
# No neural nets; purely tabular regrets & average strategy.
# Used as a ground-truth baseline to debug the PyTorch Deep CFR.

import collections
import numpy as np

import pyspiel
from open_spiel.python import policy as py_policy
from open_spiel.python.algorithms import expected_game_score


class TabularDeepCFR:
    """A simple tabular CFR solver with Deep-CFR-like outer interface.

    - Keeps tabular regrets R[I][a] and strategy sums S[I][a].
    - Uses full-tree CFR updates (no sampling).
    """

    def __init__(self, game, num_iterations=1000, log_freq=100, logger=print):
        self._game = game
        self._num_players = game.num_players()
        self._num_actions = game.num_distinct_actions()
        self._root = game.new_initial_state()

        # regrets[player][info_state][action] -> float
        self._regrets = [
            collections.defaultdict(lambda: np.zeros(self._num_actions, dtype=np.float64))
            for _ in range(self._num_players)
        ]

        # strategy_sum[player][info_state][action] -> float
        self._strategy_sum = [
            collections.defaultdict(lambda: np.zeros(self._num_actions, dtype=np.float64))
            for _ in range(self._num_players)
        ]

        self._num_iterations = num_iterations
        self._log_freq = log_freq
        self._logger = logger

        # For logging trajectories
        self._iters = []
        self._conv_hist = []
        self._v0_hist = []
        self._v1_hist = []

    # ---------- CFR core ----------

    def _get_strategy(self, player, info_state_key, legal_actions, realization_weight):
        """Compute current strategy via regret matching, and accumulate avg-strategy."""
        regrets = self._regrets[player][info_state_key]

        # regret-matching on legal actions
        positive_regrets = np.maximum(regrets, 0.0)
        sum_positive = positive_regrets[legal_actions].sum()

        strategy = np.zeros(self._num_actions, dtype=np.float64)
        if sum_positive > 0:
            for a in legal_actions:
                strategy[a] = positive_regrets[a] / sum_positive
        else:
            # uniform over legal actions
            for a in legal_actions:
                strategy[a] = 1.0 / len(legal_actions)

        # Accumulate average strategy: weighted by reach prob of this player
        self._strategy_sum[player][info_state_key] += realization_weight * strategy

        return strategy

    def _cfr(self, state, reach_probs):
        """One full-tree CFR recursion.

        Args:
            state: pyspiel.State
            reach_probs: np.array[num_players], reach prob for each player

        Returns:
            util: np.array[num_players], counterfactual values from this state.
        """
        if state.is_terminal():
            returns = np.array(state.returns(), dtype=np.float64)
            return returns

        if state.is_chance_node():
            # Enumerate all chance outcomes (small game, so it's fine).
            total_util = np.zeros(self._num_players, dtype=np.float64)
            for action, prob in state.chance_outcomes():
                child = state.child(action)
                child_util = self._cfr(child, reach_probs)
                total_util += prob * child_util
            return total_util

        current_player = state.current_player()
        legal_actions = state.legal_actions(current_player)

        info_state_key = state.information_state_string(current_player)

        # Get current policy at this infoset via regret matching
        strategy = self._get_strategy(
            current_player,
            info_state_key,
            legal_actions,
            realization_weight=reach_probs[current_player],
        )

        # util[a] = utility vector if current_player plays action a
        util = {}
        node_util = np.zeros(self._num_players, dtype=np.float64)

        for a in legal_actions:
            new_reach = reach_probs.copy()
            new_reach[current_player] *= strategy[a]
            child_state = state.child(a)
            child_util = self._cfr(child_state, new_reach)
            util[a] = child_util
            node_util += strategy[a] * child_util

        # Update regrets for current_player
        # CFR uses opponent reach prob for weighting
        opp_reach = 1.0
        for p in range(self._num_players):
            if p != current_player:
                opp_reach *= reach_probs[p]

        regrets = self._regrets[current_player][info_state_key]
        for a in legal_actions:
            regrets[a] += opp_reach * (util[a][current_player] - node_util[current_player])

        return node_util

    # ---------- Policy interface ----------

    def _average_policy_action_prob(self, state, player_id=None):
        """Callable in the form expected by tabular_policy_from_callable."""
        del player_id
        current_player = state.current_player()
        legal_actions = state.legal_actions(current_player)
        info_state_key = state.information_state_string(current_player)

        strat_sum = self._strategy_sum[current_player].get(info_state_key, None)
        probs = np.zeros(self._num_actions, dtype=np.float64)

        if strat_sum is None or strat_sum[legal_actions].sum() <= 0:
            # Fallback: uniform over legal actions
            for a in legal_actions:
                probs[a] = 1.0 / len(legal_actions)
        else:
            total = strat_sum[legal_actions].sum()
            for a in legal_actions:
                probs[a] = strat_sum[a] / total

        return {a: probs[a] for a in legal_actions}

    # ---------- Solve & logging ----------

    def solve(self):
        """Run CFR for num_iterations and log NashConv/value trajectory."""
        for it in range(1, self._num_iterations + 1):
            # Start CFR from root with all players' reach prob = 1
            self._cfr(self._root, np.ones(self._num_players, dtype=np.float64))

            if it == 1 or it % self._log_freq == 0:
                avg_policy = py_policy.tabular_policy_from_callable(
                    self._game, self._average_policy_action_prob
                )
                pyspiel_policy = py_policy.python_policy_to_pyspiel_policy(avg_policy)
                conv = pyspiel.nash_conv(self._game, pyspiel_policy)

                values = expected_game_score.policy_value(
                    self._game.new_initial_state(), [avg_policy, avg_policy]
                )

                self._iters.append(it)
                self._conv_hist.append(conv)
                self._v0_hist.append(values[0])
                self._v1_hist.append(values[1])

                self._logger(
                    f"[TabDeepCFR][iter {it}/{self._num_iterations}] "
                    f"NashConv = {conv:.6f}, "
                    f"values = ({values[0]:.4f}, {values[1]:.4f})"
                )

        return self._iters, self._conv_hist, self._v0_hist, self._v1_hist


def main():
    game_name = "kuhn_poker"
    game = pyspiel.load_game(game_name)

    solver = TabularDeepCFR(
        game,
        num_iterations=2000,  # 你可以改成 10000 看更平滑
        log_freq=200,
        logger=print,
    )
    iters, conv_hist, v0_hist, v1_hist = solver.solve()

    print("\nTrajectory (every log_freq iters):")
    for i, c, v0, v1 in zip(iters, conv_hist, v0_hist, v1_hist):
        print(
            f"  iter={i:4d}, NashConv={c:.6f}, "
            f"v0={v0:.4f} (target -0.0556), v1={v1:.4f} (target 0.0556)"
        )


if __name__ == "__main__":
    main()