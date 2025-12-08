import pyspiel
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python import policy



def policy_to_mapping(game, policy):
    from collections import deque
    mapping = {}
    q = deque([game.new_initial_state()])
    seen = set()

    while q:
        state = q.popleft()
        if state.is_terminal():
            continue
        if state.is_chance_node():
            for a, _ in state.chance_outcomes():
                child = state.child(a)
                h = child.history_str()
                if h not in seen:
                    seen.add(h); q.append(child)
            continue
        if state.is_simultaneous_node():
            continue  # Kuhn/Leduc 用不到

        info_state = state.information_state_string()
        if info_state not in mapping:
            ap = policy.action_probabilities(state)
            mapping[info_state] = [(a, float(p)) for a, p in ap.items()]

        for a in state.legal_actions():
            child = state.child(a)
            h = child.history_str()
            if h not in seen:
                seen.add(h); q.append(child)

    return mapping

def run_tabular_cfr(game_name="kuhn_poker", num_iters=20000):
    game = pyspiel.load_game(game_name)
    solver = cfr.CFRSolver(game)

    print(f"Running tabular CFR on {game_name} for {num_iters} iterations...")
    for i in range(num_iters):
        solver.evaluate_and_update_policy()
        if (i + 1) % 5000 == 0:
            print(f"  Iteration {i + 1}/{num_iters}")

    avg_policy = solver.average_policy()
    conv = exploitability.nash_conv(game, avg_policy) #pyspiel

    pyspiel_policy = policy.python_policy_to_pyspiel_policy(avg_policy)
    # mapping = policy_to_mapping(game, avg_policy)
    conv_cpp = pyspiel.nash_conv(game, pyspiel_policy)


    print(f"\n[Tabular CFR] NashConv after {num_iters} iters: {conv}")
    print(f"\n[Tabular CFR C++] NashConv after {num_iters} iters: {conv_cpp}")

    average_policy = policy.tabular_policy_from_callable(game, solver.average_policy)
    average_policy_values = expected_game_score.policy_value(
        game.new_initial_state(), [average_policy] * 2)
    print("[Tabular CFR] Computed player 0 value: %.2f (expected: %.2f)."%(average_policy_values[0], -1 / 18))
    print("[Tabular CFR] Computed player 1 value: %.2f (expected: %.2f)."%(average_policy_values[1], 1 / 18))
    
    return game, avg_policy, conv, conv_cpp


def random_policy_nashconv(game):
    # Uniform random policy over legal actions
    rand_pi = policy.UniformRandomPolicy(game)
    conv = exploitability.nash_conv(game, rand_pi) #pyspiel
    mapping = policy_to_mapping(game, rand_pi)
    conv_cpp = pyspiel.nash_conv(game, mapping)
    print(f"\n[Uniform random policy] NashConv: {conv}")
    print(f"\n[Uniform random policy C++] NashConv: {conv_cpp}")
    average_policy = policy.tabular_policy_from_callable(game, rand_pi)
    average_policy_values = expected_game_score.policy_value(
        game.new_initial_state(), [average_policy] * 2)
    print("[Uniform random policy] Computed player 0 value: %.2f (expected: %.2f)."%(average_policy_values[0], -1 / 18))
    print("[Uniform random policy] Computed player 1 value: %.2f (expected: %.2f)."%(average_policy_values[1], 1 / 18))
    return conv, conv_cpp


if __name__ == "__main__":
    # 1) Tabular CFR policy
    game, cfr_policy, cfr_conv, cfr_conv_cpp = run_tabular_cfr("kuhn_poker", num_iters=20000)

    # 2) Uniform random policy on the same game
    rand_conv, rand_conv_cpp = random_policy_nashconv(game)

    print("\nSummary:")
    print(f"  NashConv (Tabular CFR)        : {cfr_conv}")
    print(f"  NashConv (Uniform random)     : {rand_conv}")
    print(f"  NashConv (Tabular CFR) C++    : {cfr_conv_cpp}")
    print(f"  NashConv (Uniform random) C++ : {rand_conv_cpp}")