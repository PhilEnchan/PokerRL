# Copyright (c) 2019 Eric Steinberger


import unittest
from unittest import TestCase

import numpy as np

from PokerRL.game._.tree.PublicTree import PublicTree
from PokerRL.game.games import StandardLeduc, DiscretizedNLLeduc,Numeral211,LimitHoldem
from PokerRL.game.wrappers import HistoryEnvBuilder

# pytest test/game/test_tree.py -k TestGameTree 2>&1 | tee error.txt
class TestGameTree(TestCase):

    # def test_impossible_1(self):
    #     from PokerRL.game.PokerRange import PokerRange
    #     env_args = Numeral211.ARGS_CLS(n_seats=2,
    #                                     starting_stack_sizes_list=[500, 500]
    #                                     )
    #     env_bldr = HistoryEnvBuilder(env_cls=Numeral211, env_args=env_args)
    #     env = env_bldr.get_new_env(is_evaluating=True)
    #     boards_2d = np.array([[0, 1], [2, 3]], dtype=np.int8)
    #     for _ in range(0,10000):
    #         PokerRange.get_possible_range_idxs(env, env_bldr.lut_holder, boards_2d)
        # print("PokerRange.get_possible_range_idxs(env, env_bldr.lut_holder, boards_2d)", PokerRange.get_possible_range_idxs(env, env_bldr.lut_holder, boards_2d))
        # assert 0

    # def test_impossible_2(self):
    #     env_args = Numeral211.ARGS_CLS(n_seats=2,
    #                                     starting_stack_sizes_list=[500, 500]
    #                                     )
    #     env_bldr = HistoryEnvBuilder(env_cls=Numeral211, env_args=env_args)
    #     board_2d = np.array([[0, 1], [2, 3]], dtype=np.int8)
    #     from itertools import chain
    #     from PokerRL.game.Poker import Poker
    #     for _ in range(0,10000):
    #         board_used_cards = env_bldr.lut_holder.get_1d_cards(board_2d)
    #         impossible_range_idxs = set()
    #         impossible_range_idxs.update(chain.from_iterable(
    #             env_bldr.lut_holder.LUT_CARD_IN_WHAT_RANGE_IDXS[c]
    #             for c in board_used_cards if c != Poker.CARD_NOT_DEALT_TOKEN_1D
    #         ))
    #         impossible_range_idxs = np.array(list(impossible_range_idxs))
        # board_used_cards = env_bldr.lut_holder.get_1d_cards(board_2d)
        # impossible_range_idxs = set()
        # impossible_range_idxs.update(chain.from_iterable(
        #     env_bldr.lut_holder.LUT_CARD_IN_WHAT_RANGE_IDXS[c]
        #     for c in board_used_cards if c != Poker.CARD_NOT_DEALT_TOKEN_1D
        # ))
        # impossible_range_idxs = np.array(list(impossible_range_idxs))
        # print("impossible_range_idxs", impossible_range_idxs)
        # assert 0

    def test_building(self):
        _get_leduc_tree()
        _get_nl_leduc_tree()
    #     _get_numeral211_tree()

    # def test_vs_env_obs(self):
    #     for game in ["limit", "nl"]:
    #         if game == "limit":
    #             env, env_args = _get_new_leduc_env()
    #             dummy_env, env_args = _get_new_leduc_env()
    #             tree = _get_leduc_tree(env_args=env_args)
    #         else:
    #             env, env_args = _get_new_nl_leduc_env()
    #             dummy_env, env_args = _get_new_nl_leduc_env()
    #             tree = _get_nl_leduc_tree(env_args=env_args)

    #         lut_holder = StandardLeduc.get_lut_holder()

    #         env.reset()
    #         dummy_env.reset()
    #         node = tree.root

    #         # RAISE .. stays preflop
    #         legal = env.get_legal_actions()
    #         a = 2
    #         assert a in legal

    #         o, r, d, i = env.step(a)
    #         node = node.children[legal.index(a)]
    #         dummy_env.load_state_dict(node.env_state)
    #         tree_o = dummy_env.get_current_obs(is_terminal=False)

    #         env.print_obs(o)
    #         env.print_obs(tree_o)
    #         assert np.array_equal(o, tree_o)

    #         # CALL .. goes flop
    #         legal = env.get_legal_actions()
    #         a = 1
    #         assert a in legal

    #         o, r, d, i = env.step(a)
    #         node = node.children[legal.index(1)]
    #         card_that_came_in_env = lut_holder.get_1d_card(env.board[0])
    #         node = node.children[card_that_came_in_env]
    #         dummy_env.load_state_dict(node.env_state)
    #         tree_o = dummy_env.get_current_obs(is_terminal=False)

    #         assert np.array_equal(o, tree_o)

    #         # RAISE .. stays flop
    #         legal = env.get_legal_actions()
    #         a = legal[-1]
    #         assert a in legal

    #         o, r, d, i = env.step(a)
    #         node = node.children[legal.index(a)]
    #         dummy_env.load_state_dict(node.env_state)
    #         tree_o = dummy_env.get_current_obs(is_terminal=False)

    #         assert np.array_equal(o, tree_o)
    #     assert 0

    def test_numeral211_env_obs_single(self):
        # env, env_args = _get_new_numeral211_env()
        # dummy_env, env_args = _get_new_numeral211_env()
        # lut_holder = Numeral211.get_lut_holder()
        
        env, env_args = _get_new_hunl_env()
        dummy_env, env_args = _get_new_hunl_env()
        lut_holder = LimitHoldem.get_lut_holder()

        env.reset()
        dummy_env.reset()

        # RAISE .. stays preflop
        legal = env.get_legal_actions()
        a = 2
        assert a in legal

        o, r, d, i = env.step(a)

        env.print_obs(o)
        assert 0

    # def test_numeral211_env_obs(self):

    #     env, env_args = _get_new_numeral211_env()
    #     dummy_env, env_args = _get_new_numeral211_env()
    #     tree = _get_numeral211_tree(env_args=env_args)

    #     lut_holder = Numeral211.get_lut_holder()

    #     env.reset()
    #     dummy_env.reset()
    #     node = tree.root

    #     # RAISE .. stays preflop
    #     legal = env.get_legal_actions()
    #     a = 2
    #     assert a in legal

    #     o, r, d, i = env.step(a)
    #     node = node.children[legal.index(a)]
    #     dummy_env.load_state_dict(node.env_state)
    #     tree_o = dummy_env.get_current_obs(is_terminal=False)

    #     env.print_obs(o)
    #     env.print_obs(tree_o)
    #     print("tree_o")
    #     print(type(tree_o))
    #     # assert np.array_equal(o, tree_o)

    #     # CALL .. goes flop
    #     legal = env.get_legal_actions()
    #     a = 1
    #     assert a in legal

    #     o, r, d, i = env.step(a)
    #     node = node.children[legal.index(1)]
    #     card_that_came_in_env = lut_holder.get_1d_card(env.board[0])
    #     node = node.children[card_that_came_in_env]
    #     dummy_env.load_state_dict(node.env_state)
    #     tree_o = dummy_env.get_current_obs(is_terminal=False)

    #     # assert np.array_equal(o, tree_o)

    #     # RAISE .. stays flop
    #     legal = env.get_legal_actions()
    #     a = legal[-1]
    #     assert a in legal

    #     o, r, d, i = env.step(a)
    #     node = node.children[legal.index(a)]
    #     dummy_env.load_state_dict(node.env_state)
    #     tree_o = dummy_env.get_current_obs(is_terminal=False)

    #     # assert np.array_equal(o, tree_o)
    #     # assert 0


def _get_leduc_tree(env_args=None):
    if env_args is None:
        env_args = StandardLeduc.ARGS_CLS(n_seats=2,
                                          starting_stack_sizes_list=[100, 100]
                                          )

    env_bldr = HistoryEnvBuilder(env_cls=StandardLeduc, env_args=env_args)

    _tree = PublicTree(
        env_bldr=env_bldr,
        stack_size=env_args.starting_stack_sizes_list,
        stop_at_street=None
    )
    print("env_args.starting_stack_sizes_list", env_args.starting_stack_sizes_list)

    _tree.build_tree()

    for p in range(env_bldr.N_SEATS):
        _tree.fill_uniform_random()
    _tree.compute_ev()

    _tree.export_to_file()
    print("Tree with stack size", _tree.stack_size, "has", _tree.n_nodes, "nodes out of which", _tree.n_nonterm,
          "are non-terminal.")
    print(np.mean(_tree.root.exploitability) * env_bldr.env_cls.EV_NORMALIZER)

    return _tree


def _get_nl_leduc_tree(env_args=None):
    if env_args is None:
        env_args = DiscretizedNLLeduc.ARGS_CLS(n_seats=2,
                                               starting_stack_sizes_list=[1000, 1000],
                                               bet_sizes_list_as_frac_of_pot=[1.0]
                                               )

    env_bldr = HistoryEnvBuilder(env_cls=DiscretizedNLLeduc, env_args=env_args)

    _tree = PublicTree(
        env_bldr=env_bldr,
        stack_size=env_args.starting_stack_sizes_list,
        stop_at_street=None,
    )

    _tree.build_tree()

    for p in range(env_bldr.N_SEATS):
        _tree.fill_uniform_random()
    _tree.compute_ev()

    _tree.export_to_file()
    print("Tree with stack size", _tree.stack_size, "has", _tree.n_nodes, "nodes out of which", _tree.n_nonterm,
          "are non-terminal.")
    print(np.mean(_tree.root.exploitability) * env_bldr.env_cls.EV_NORMALIZER)

    return _tree


def _get_new_leduc_env(env_args=None):
    if env_args is None:
        env_args = StandardLeduc.ARGS_CLS(n_seats=2,
                                          starting_stack_sizes_list=[150, 150],
                                          )
    return StandardLeduc(env_args=env_args, is_evaluating=True, lut_holder=StandardLeduc.get_lut_holder()), env_args


def _get_new_nl_leduc_env(env_args=None):
    if env_args is None:
        env_args = DiscretizedNLLeduc.ARGS_CLS(n_seats=2,
                                               bet_sizes_list_as_frac_of_pot=[1.0, 1000.0]
                                               )
    return DiscretizedNLLeduc(env_args=env_args, is_evaluating=True,
                              lut_holder=DiscretizedNLLeduc.get_lut_holder()), env_args

def _get_new_numeral211_env(env_args=None):
    if env_args is None:
        env_args = Numeral211.ARGS_CLS(n_seats=2,
                                       starting_stack_sizes_list=[500, 500],
                                       )
    return Numeral211(env_args=env_args, is_evaluating=True, lut_holder=Numeral211.get_lut_holder()), env_args

def _get_new_hunl_env(env_args=None):
    if env_args is None:
        env_args = LimitHoldem.ARGS_CLS(n_seats=2,
                                       starting_stack_sizes_list=[500, 500],
                                       )
    return LimitHoldem(env_args=env_args, is_evaluating=True, lut_holder=LimitHoldem.get_lut_holder()), env_args

def _get_numeral211_tree(env_args=None):
    if env_args is None:
        env_args = Numeral211.ARGS_CLS(n_seats=2,
                                       starting_stack_sizes_list=[500, 500]
                                       )

    env_bldr = HistoryEnvBuilder(env_cls=Numeral211, env_args=env_args)

    _tree = PublicTree(
        env_bldr=env_bldr,
        stack_size=env_args.starting_stack_sizes_list,
        stop_at_street=None
    )

    _tree.build_tree()

    for p in range(env_bldr.N_SEATS):
        _tree.fill_uniform_random()
    _tree.compute_ev()

    _tree.export_to_file()
    print("Tree with stack size", _tree.stack_size, "has", _tree.n_nodes, "nodes out of which", _tree.n_nonterm,
          "are non-terminal.")
    print(np.mean(_tree.root.exploitability) * env_bldr.env_cls.EV_NORMALIZER)

    return _tree

if __name__ == '__main__':
    unittest.main()
