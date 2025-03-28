# Copyright (c) 2019 Eric Steinberger


import os
import ctypes
from os.path import join as ospj

import numpy as np

from PokerRL._.CppWrapper import CppWrapper
from PokerRL.game.Poker import Poker
from PokerRL.game._.rl_env.game_rules import Numeral211Rules


class CppLibNumeral211Luts(CppWrapper):
    def __init__(self):
        super().__init__(path_to_dll=ospj(os.path.dirname(os.path.realpath(__file__)),
                                          "lib_numeral211_luts.so"))
        self._clib.batch_get_1d_card.argtypes = [self.ARR_2D_ARG_TYPE, ctypes.c_int, ctypes.c_void_p]
        self._clib.batch_get_1d_card.restype = None

        self._clib.batch_get_2d_card.argtypes = [ctypes.c_void_p, ctypes.c_int, self.ARR_2D_ARG_TYPE]
        self._clib.batch_get_2d_card.restype = None

        self._clib.get_hole_card_2_idx_lut.argtypes = [self.ARR_2D_ARG_TYPE]
        self._clib.get_hole_card_2_idx_lut.restype = None

        self._clib.get_idx_2_hole_card_lut.argtypes = [self.ARR_2D_ARG_TYPE]
        self._clib.get_idx_2_hole_card_lut.restype = None

    def batch_get_1d_card(self, card_2d_arr):
        """
        Args:
            card_2d_arr (np.ndarray): shape=(N, 2), dtype=np.int8
                2D representations of the input cards. Each row contains [rank, suit].

        Returns:
            np.ndarray: shape=(N,), dtype=np.int8
                1D representation of the input cards.
        """
        N = card_2d_arr.shape[0]
        card_1d_arr = np.empty(shape=(N,), dtype=np.int8)

        # 调用 C++ 函数
        self._clib.batch_get_1d_card(self.np_2d_arr_to_c(card_2d_arr), N, self.np_1d_arr_to_c(card_1d_arr))

        return card_1d_arr
        


    def batch_get_2d_card(self, card_1d_arr):
        """
        Args:
            card_1d_arr (np.ndarray): Array of 1D card representations.

        Returns:
            np.ndarray(shape=(N, 2), dtype=np.int8): 2D representations of the input cards.
        """
        N = card_1d_arr.shape[0]
        card_2d_arr = np.empty(shape=(N, 2), dtype=np.int8)
        self._clib.batch_get_2d_card(self.np_1d_arr_to_c(card_1d_arr), N, self.np_2d_arr_to_c(card_2d_arr))
        return card_2d_arr
    
    def get_idx_2_hole_card_lut(self):
        lut = np.full(shape=(Numeral211Rules.RANGE_SIZE, 2), fill_value=-2, dtype=np.int8)
        self._clib.get_idx_2_hole_card_lut(self.np_2d_arr_to_c(lut))  # fills it
        return lut

    def get_hole_card_2_idx_lut(self):
        lut = np.full(shape=(Numeral211Rules.N_CARDS_IN_DECK, Numeral211Rules.N_CARDS_IN_DECK),
                      fill_value=-2, dtype=np.int16)
        self._clib.get_hole_card_2_idx_lut(self.np_2d_arr_to_c(lut))  # fills it
        return lut