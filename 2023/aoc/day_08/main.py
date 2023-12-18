import itertools as it
import re
from timeit import timeit
from functools import reduce

import numpy as np
import torch

from . import inputs


def parse(input):
    moves, _, *states = input.splitlines()

    move_map = {"L": 0, "R": 1}
    moves = np.array([move_map[move] for move in moves])

    def parse_state(state):
        match = re.search(r"(...) = \((...), (...)\)", state)
        if not match:
            return None

        return match.group(1), match.group(2), match.group(3)

    states = {
        parsed[0]: (parsed[1], parsed[2])
        for state in states
        if (parsed := parse_state(state))
    }

    return moves, states


def input_to_matrix_representation(input):
    moves, states = parse(input)
    move_indexer = torch.from_numpy(moves)

    num_states = len(states)
    state_dict = {name: ix for ix, name in enumerate(states)}
    state_matrix = torch.stack(
        [
            torch.tensor([state_dict[target] for target in state], dtype=torch.long)
            for state in states.values()
        ],
        dim=0,
    )
    state_transition_matrix = torch.zeros([2, num_states, num_states])
    for state_ix, (left, right) in enumerate(state_matrix):
        state_transition_matrix[0, state_ix, left] = 1
        state_transition_matrix[1, state_ix, right] = 1

    single_loop = reduce(
        lambda a, b: a @ b, [state_transition_matrix[move] for move in move_indexer]
    )

    return move_indexer, single_loop, state_dict


def state_vector(state_name, state_dict):
    state = torch.zeros([1, len(state_dict)])
    state[0, state_dict[state_name]] = 1
    return state


def first_task(input, device="cuda"):
    (
        move_indexer,
        single_loop,
        state_dict,
    ) = input_to_matrix_representation(input)

    single_loop = single_loop.to(device)
    end_state = state_vector("ZZZ", state_dict).to(device)
    current_state = state_vector("AAA", state_dict).to(device)

    for steps in it.count(0):
        if (current_state * end_state).sum() > 0:
            return steps * len(move_indexer)
        current_state = current_state @ single_loop


def second_task(input, device="cuda"):
    (
        move_indexer,
        single_loop,
        state_dict,
    ) = input_to_matrix_representation(input)

    end_state_mask = torch.stack(
        [state_vector(state, state_dict) for state in state_dict if state.endswith("Z")]
    ).sum(dim=0)
    current_states = torch.stack(
        [state_vector(state, state_dict) for state in state_dict if state.endswith("A")]
    )

    single_loop = single_loop.to(device)
    end_state_mask = end_state_mask.to(device)
    current_states = current_states.to(device)

    steps_until_end = {}

    for steps in it.count(0):
        states_at_end = torch.nonzero((current_states * end_state_mask).sum(dim=-1))
        if len(states_at_end) > 0:
            for state in states_at_end:
                state = state[..., 0].item()
                if state not in steps_until_end:
                    steps_until_end[state] = steps

            if len(steps_until_end) == len(current_states):
                break

        current_states = current_states @ single_loop

    num_loops = np.lcm.reduce(list(steps_until_end.values()))
    return num_loops * len(move_indexer)


if __name__ == "__main__":
    print(first_task(inputs.full()[-1]))
    print(second_task(inputs.full()[-1]))
    with torch.no_grad():
        print("CPU: ", timeit(lambda: second_task(inputs.full()[-1], "cpu"), number=5))
        print("GPU: ", timeit(lambda: second_task(inputs.full()[-1], "cuda"), number=5))
