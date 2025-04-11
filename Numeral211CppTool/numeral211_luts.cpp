// g++ -shared -o PokerRL/game/_/cpp_wrappers/lib_numeral211_luts.so -fPIC Numeral211CppTool/numeral211_luts.cpp -std=c++17

#include <cstdint>
#include <algorithm>

#include "numeral211.h"

extern "C" void batch_get_2d_card(void* card_1d_arr, int num_cards, void* card_2d_arr) {
    const int8_t* cards_1d = static_cast<const int8_t*>(card_1d_arr);
    int8_t** cards_2d = static_cast<int8_t**>(card_2d_arr); // 这里是 int8_t**，因为 Python 传递的是指针数组

    for (int i = 0; i < num_cards; ++i) {
        cards_2d[i][0] = cards_1d[i] / numeral211::SUITS;
        cards_2d[i][1] = cards_1d[i] % numeral211::SUITS;
    }
}

extern "C" void batch_get_1d_card(void* card_2d_arr, int num_cards, void* card_1d_arr) {
    const int8_t** cards_2d = static_cast<const int8_t**>(card_2d_arr); // 2D数组指针
    int8_t* cards_1d = static_cast<int8_t*>(card_1d_arr);   // 1D数组指针

    for (int i = 0; i < num_cards; ++i) {
        cards_1d[i] = cards_2d[i][0] * numeral211::SUITS + cards_2d[i][1];
    }
}

extern "C" void get_idx_2_hole_card_lut(void* input) {
    int8_t** lut = static_cast<int8_t**>(input);
    int index = 0;
    for (int8_t c1 = 0; c1 < numeral211::CARDS_IN_DECK; ++c1) {
        for (int8_t c2 = c1 + 1; c2 < numeral211::CARDS_IN_DECK; ++c2) {
            lut[index][0] = c1;
            lut[index][1] = c2;
            ++index;
        }
    }
}

extern "C" void get_hole_card_2_idx_lut(void* input) {
    int16_t** lut = static_cast<int16_t**>(input);
    // 初始化 LUT，确保所有值都设置为 -2
    int index = 0;
    for (int c1 = 0; c1 < numeral211::CARDS_IN_DECK; ++c1) {
        std::fill(lut[c1], lut[c1] + numeral211::CARDS_IN_DECK, static_cast<int16_t>(-2));
        for (int c2 = c1 + 1; c2 < numeral211::CARDS_IN_DECK; ++c2) {
            lut[c1][c2] = index;
            ++index;
        }
    }
}