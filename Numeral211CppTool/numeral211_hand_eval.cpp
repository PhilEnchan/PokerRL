// print(lh.LUT_IDX_2_HOLE_CARDS)
// print(lh.LUT_1DCARD_2_2DCARD)

// assert 0

// [[ 0  1]
//  [ 0  2]
//  [ 0  3]
//  ...
//  [49 50]
//  [49 51]
//  [50 51]]
// [[ 0  0]
//  [ 0  1]
//  [ 0  2]
//  [ 0  3]
//  [ 1  0]
//  [ 1  1]
//  [ 1  2]
//  [ 1  3]
//  [ 2  0]
//  [ 2  1]
//  [ 2  2]
//  [ 2  3]
//  [ 3  0]
//  [ 3  1]
//  [ 3  2]
//  [ 3  3]
//  [ 4  0]
//  [ 4  1]
//  [ 4  2]
//  [ 4  3]
//  [ 5  0]
//  [ 5  1]
//  [ 5  2]
//  [ 5  3]
//  [ 6  0]
//  [ 6  1]
//  [ 6  2]
//  [ 6  3]
//  [ 7  0]
//  [ 7  1]
//  [ 7  2]
//  [ 7  3]
//  [ 8  0]
//  [ 8  1]
//  [ 8  2]
//  [ 8  3]
//  [ 9  0]
//  [ 9  1]
//  [ 9  2]
//  [ 9  3]
//  [10  0]
//  [10  1]
//  [10  2]
//  [10  3]
//  [11  0]
//  [11  1]
//  [11  2]
//  [11  3]
//  [12  0]
//  [12  1]
//  [12  2]
//  [12  3]]

// g++ -shared -o PokerRL/game/_/cpp_wrappers/lib_numeral211_hand_eval.so -fPIC Numeral211CppTool/numeral211_hand_eval.cpp -std=c++17 -pthread
// g++ -shared -o PokerRL/game/_/cpp_wrappers/lib_numeral211_hand_eval.so -fPIC Numeral211CppTool/numeral211_hand_eval.cpp -std=c++17 -pthread -DDEBUG


#include <mutex>
#include <iostream>
#include <fstream>
#include <array>
#include <cassert>
#include <filesystem>

#include "numeral211.h"

const int LUT_LENGTH = 15593606;
static std::array<int32_t, LUT_LENGTH> lookup;
std::once_flag flag;
const int8_t CARD_NOT_DEALT_TOKEN = -1;
std::string filePath;

void load_lut() {
    // 获取当前工作目录
    // std::filesystem::path currentPath = std::filesystem::current_path();
    // // 打印当前工作目录
    // std::cout << "Current directory: " << currentPath << std::endl;
    std::cout << "file path: " << filePath << std::endl;
    uint16_t num = 1;
    bool isLittleEndian = *reinterpret_cast<uint8_t*>(&num) == 1;
    std::cout<<"isLittleEndian: "<< isLittleEndian<<std::endl;
    
    // const std::string filePath = "RhodeIslandHandRanks.dat";
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Couldn't open file: " + filePath);
    }
    
    // 检查文件大小
    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    if (fileSize != static_cast<std::streamsize>(LUT_LENGTH * sizeof(int32_t))) {
        throw std::runtime_error("File size does not match expected data size.");
    }

    // 读取数据到数组
    file.read(reinterpret_cast<char*>(lookup.data()), fileSize);
    // 检查读取是否成功
    if (!file) {
        throw std::runtime_error("Failed to read file data.");
    }
}

#ifdef DEBUG
#include <signal.h>
const int8_t** last_hand_2d = nullptr;
const int8_t** last_board_2d = nullptr;
const int8_t** last_boards_1d = nullptr;
const int8_t** last_hands_1d = nullptr;
int last_num_boards, last_num_hands, last_board_len, last_hand_len;

// 处理 SIGSEGV 信号的函数
void segfault_handler(int signum) {
    std::ofstream fs("lib_numeral211_hand_eval_error.log", std::ios::app); // 追加模式，防止覆盖
    if (!fs.is_open()) {
        std::cerr << "Error: Unable to open segfault.log!" << std::endl;
        exit(1);
    }
    if (signum == SIGSEGV) {
        fs << "Segmentation fault (SIGSEGV) detected!" << std::endl;
    } else if (signum == SIGABRT) {
        fs << "Aborted (SIGABRT) detected!" << std::endl;
    }

    if (last_hand_2d) {
        fs << "Last hand_2d: ";
        for (int k = 0; k< last_hand_len; ++k) {
            fs << "("<<static_cast<int>(last_hand_2d[k][0]) << ", " << static_cast<int>(last_hand_2d[k][1]) << "); ";
        }
        fs << std::endl;
    }
    if (last_board_2d) {
        fs << "Last board_2d: ";
        for (int k = 0; k< last_board_len; ++k) {
            fs << "("<<static_cast<int>(last_board_2d[k][0]) << ", " << static_cast<int>(last_board_2d[k][1]) << "); ";
        }
        fs << std::endl;
    }
    if (last_boards_1d) {
        fs << "Last boards_1d: ";
        for (int i = 0; i < last_num_boards; ++i) {
            fs << "[";
            for (int k = 0; k< last_board_len; ++k) {
                fs << static_cast<int>(last_boards_1d[i][k]) << ", ";
            }
            fs << "]; ";
        }
        fs << std::endl;
    }
    if (last_hands_1d) {
        fs << "Last hands_1d: ";
        for (int i = 0; i < last_num_hands; ++i) {
            fs << "[";
            for (int k = 0; k< last_hand_len; ++k) {
                fs << static_cast<int>(last_hands_1d[i][k]) << ", ";
            }
            fs << "]; ";
        }
        fs << std::endl;
    }
    fs << "dat file: " << filePath << std::endl;
    exit(1);
}
#endif

// 导出 signal 处理函数，供 Python 端调用
extern "C" void setup_signal_handler(const char* path) {
    filePath = std::string(path);
#ifdef DEBUG
    signal(SIGSEGV, segfault_handler);
    signal(SIGABRT, segfault_handler);
#endif
}


extern "C" int32_t get_hand_rank_numeral211(void* hand_2d_raw, void* board_2d_raw, int hand_len, int board_len) {
    const int8_t** hand_2d = static_cast<const int8_t**>(hand_2d_raw);
    const int8_t** board_2d = static_cast<const int8_t**>(board_2d_raw);
#ifdef DEBUG
    last_board_2d = board_2d;
    last_hand_2d = hand_2d;
    last_board_len = board_len;
    last_hand_len = hand_len;
#endif

    assert(hand_len == 2 && board_len == 2);


    std::call_once(flag, load_lut);

    int32_t value = 53;
    bool not_dealt = false;
    for (int k = 0; k < hand_len; ++k) {
        if (hand_2d[k][0] == CARD_NOT_DEALT_TOKEN || hand_2d[k][1] == CARD_NOT_DEALT_TOKEN) {
            not_dealt = true;
            break;
        }
        int8_t c = hand_2d[k][0] * numeral211::SUITS + hand_2d[k][1];
        value = lookup[value + 13 + c];
    }

    for (int k = 0; k < board_len; ++k) {
        if (not_dealt || board_2d[k][0] == CARD_NOT_DEALT_TOKEN || board_2d[k][1] == CARD_NOT_DEALT_TOKEN) {
            value = 0;
            not_dealt = true;
            break;
        }
        int8_t c = board_2d[k][0] * numeral211::SUITS + board_2d[k][1];
        value = lookup[value + 13 + c];
    }

    value = lookup[value];
    if (value == 0)
    { value = -1;}

#ifdef DEBUG
    last_board_2d = nullptr;
    last_hand_2d = nullptr;
#endif
    return value;
}

extern "C" void get_hand_rank_given_boards_dim0_hands_dim1_numeral211(void* boards_1d_raw, void* hands_1d_raw, int num_boards, int num_hands, int board_len, int hand_len, void* hand_ranks_raw) {
    const int8_t** boards_1d = static_cast<const int8_t**>(boards_1d_raw);
    const int8_t** hands_1d = static_cast<const int8_t**>(hands_1d_raw);
#ifdef DEBUG
    last_boards_1d = boards_1d;
    last_hands_1d = hands_1d;
    last_num_boards = num_boards;
    last_num_hands = num_hands;
    last_board_len = board_len;
    last_hand_len = hand_len;
#endif

    assert(board_len == 2 && hand_len == 2);

    int32_t** hand_ranks = static_cast<int32_t**>(hand_ranks_raw);

    std::call_once(flag, load_lut);

    for (int i = 0; i < num_boards; ++i) {
        int board_value = 53;
        bool not_dealt = false;
        for (int k = 0; k < board_len; ++k) {
            if (boards_1d[i][k] == CARD_NOT_DEALT_TOKEN) {
                not_dealt = true;
                break;
            }
            board_value = lookup[board_value + 13 + boards_1d[i][k]];
        }

        for (int j = 0; j < num_hands; ++j) {
            int32_t& value = hand_ranks[i][j];
            value = board_value;
            
            for (int k = 0; k < hand_len; ++k) {
                if (not_dealt || hands_1d[j][k] == CARD_NOT_DEALT_TOKEN) {
                    value = 0;
                    not_dealt = true;
                    break;
                }
                value = lookup[value + 13 + hands_1d[j][k]];
            }

            value = lookup[value];
            if (value == 0)
            { value = -1;}
        }
    }
#ifdef DEBUG
    last_boards_1d = nullptr;
    last_hands_1d = nullptr;
#endif
}

// // g++ -o PokerRL/game/_/cpp_wrappers/numeral211_hand_eval Numeral211CppTool/numeral211_hand_eval.cpp -pthread

// int main() {
//     std::call_once(flag, load_lut);
//     return 0;
// }