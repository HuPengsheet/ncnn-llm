#include <vector>
#include <cstdint>
#include <string>
#include <map>
#include <set>
#include <queue>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <functional>
#include <memory>
#include <locale>
#include <codecvt>


struct Tokenizer {
    enum TokenizerType {
        BPE = 0,
        NORMAL = 1,
        QWEN = 2,
        GLM = 3
    };

    struct TrieNode {
        int tokenId;
        float score;
        std::map <int, TrieNode*> next;
        TrieNode();
    };
    struct Symbol {
        TrieNode *node;
        char *s;
        int pos, len;
        int prev, next;
        int fixId;

        Symbol (Tokenizer::TrieNode *node,
                char *s, int pos, int len,
                int prev, int next, int fixId) {
            this->node = node;
            this->s = s;
            this->pos = pos;
            this->len = len;
            this->prev = prev;
            this->next = next;
            this->fixId = fixId;
        }
    };
    struct SymbolPairs {
        float score;
        int l, r, size;

        SymbolPairs(float score, int l, int r, int size) {
            this->score = score;
            this->l = l;
            this->r = r;
            this->size = size;
        }
    };

    friend bool operator < (const SymbolPairs &a, const SymbolPairs &b) {
        return a.score < b.score || (a.score == b.score && a.l > b.l);
    }

    TrieNode *root;

    TrieNode *specialRoot = nullptr;

    TokenizerType type = TokenizerType::BPE;

    bool addDummyPrefix = true;   // 是否在首位添加空格
    bool removeExtraWhitespaces = true;   // 是否将多个空格合并为一个
    bool byteAsChar = false;  // 是否将byte变为展示字符

    std::unordered_map <int, std::string> tokenToStringDict;
    std::unordered_map <int, float> tokenToScoreDict;
    std::unordered_map <std::string, int> stringToTokenDict;
    std::vector <std::string> specialTokens;

    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::unordered_map <wchar_t, wchar_t> byteCharDict;
    std::unordered_map <wchar_t, wchar_t> charByteDict;

    Tokenizer ();

    ~Tokenizer();

    void Clear(); // 清空分词器

    void TryMergePairs(std::vector<Symbol> &symbols, int l, int r, std::priority_queue <SymbolPairs> &q); // 插入备选symbol

    int GetRank(std::vector<Symbol> &symbols,  std::vector<std::pair<int, int>> &partitions, int idx, int skip);

    void Insert(const std::string &s, int tokenId, float score = 1.0f); // 插入一个token

    void SetSpecialTokens(const std::map <std::string, int> &specialTokens); // 设置需要优先处理的特殊token

    std::string Normalize(const std::string &ori); // 字符规范化

    std::vector<float> Encode(const std::string &s); // 编码

    std::string Decode(const std::vector <int> &tokens); // 解码
};
