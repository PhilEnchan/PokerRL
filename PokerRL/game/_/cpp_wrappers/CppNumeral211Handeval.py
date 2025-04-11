# Copyright (c) 2019 Eric Steinberger


import ctypes
import os
from os.path import join as ospj

import numpy as np

from PokerRL._.CppWrapper import CppWrapper
from PokerRL.game._.rl_env.game_rules import Numeral211Rules

class CppNumeral211Handeval(CppWrapper):
    def __init__(self):
        dll_path = ospj(os.path.dirname(os.path.realpath(__file__)),
                                    "lib_numeral211_hand_eval.so")
        dat_path = ospj(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))),
                                          "Numeral211CppTool/RhodeIslandHandRanks.dat")
        super().__init__(path_to_dll=dll_path)
        self._clib.get_hand_rank_numeral211.argtypes = [
            self.ARR_2D_ARG_TYPE,
            self.ARR_2D_ARG_TYPE,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        self._clib.get_hand_rank_numeral211.restype = ctypes.c_int32

        self._clib.get_hand_rank_given_boards_dim0_hands_dim1_numeral211.argtypes = [
            self.ARR_2D_ARG_TYPE,
            self.ARR_2D_ARG_TYPE,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            self.ARR_2D_ARG_TYPE
        ]
        self._clib.get_hand_rank_given_boards_dim0_hands_dim1_numeral211.restype = None
        
        self._clib.setup_signal_handler.argtypes = [ctypes.c_char_p]
        self._clib.setup_signal_handler(dat_path.encode('utf-8'))
    
    def get_hand_rank_numeral211(self, hand_2d, board_2d):
        """
        Args:
            hand_2d (np.ndarray(shape=[2,2], dtype=int8)):      [rank, suit], [rank, suit]]
            board_2d (np.ndarray(shape=[2,2], dtype=int8)):     [rank, suit], [rank, suit], ...]

        Returns:
            int: integer representing strength of the strongest 3card hand in the 4 cards. higher is better.
        """
        return self._clib.get_hand_rank_numeral211(self.np_2d_arr_to_c(hand_2d), self.np_2d_arr_to_c(board_2d), hand_2d.shape[0], board_2d.shape[0])
    
    def get_hand_rank_all_hands_on_given_boards_numeral211(self, boards_1d, lut_holder):
        """
        Args:
            boards_1d (np.ndarray(shape=[N, 2], dtype=int8)):   [[c1, c2], [c1, c2], ...]

        Returns:
            np.ndarray(shape=[N, RANGE_SIZE], dtype=int32):     hand_rank for each possible hand; -1 for
                                                                blocked on each of the given boards
        """
        assert len(boards_1d.shape) == 2
        hand_ranks = np.full(shape=(boards_1d.shape[0], Numeral211Rules.RANGE_SIZE), fill_value=-1, dtype=np.int32)
        self._clib.get_hand_rank_given_boards_dim0_hands_dim1_numeral211(
            self.np_2d_arr_to_c(boards_1d),  # int8**
            self.np_2d_arr_to_c(lut_holder.LUT_IDX_2_HOLE_CARDS),  # int8**
            boards_1d.shape[0],  # int (number of boards)
            lut_holder.LUT_IDX_2_HOLE_CARDS.shape[0],  # int (number of hole cards)
            boards_1d.shape[1],  # int (number of cards on board)
            lut_holder.LUT_IDX_2_HOLE_CARDS.shape[1],  # int (number of cards in hole)
            self.np_2d_arr_to_c(hand_ranks),  # int32**
        )
        return hand_ranks