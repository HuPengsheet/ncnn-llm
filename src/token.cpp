#include "token.h"
#include <cstring>
#include <cmath>
#include <cfloat>
#include <thread>
#include <algorithm>

    Tokenizer::TrieNode::TrieNode() {
        this->tokenId = -999999;
    }

    Tokenizer::Tokenizer() {
        root = new TrieNode();
        int n = 0;
        wchar_t special_token = L'\x0';
        for (; special_token < L'!'; special_token++, n++) {
            byteCharDict[L'\x100' + n] = special_token;
            charByteDict[special_token] = L'\x100' + n;
        }
        for (special_token = L'\x7F'; special_token < L'\xA1'; special_token++, n++) {
            byteCharDict[L'\x100' + n] = special_token;
            charByteDict[special_token] = L'\x100' + n;
        }
        byteCharDict[L'\x100' + n++] = L'\xAD';
        charByteDict[L'\xAD'] = L'\x100' + n++;
    }

    Tokenizer::~Tokenizer() {
        Clear();
        delete root;
    }

    void Tokenizer::Clear() {
        std::vector <TrieNode*> q;
        q.push_back(root);
        for (int i = 0; i < q.size(); i++) {
            TrieNode *now = q[i];
            for (auto it : now->next) {
                q.push_back(it.second);
            }
        }
        if (specialRoot != nullptr) {
            q.push_back(specialRoot);
            for (int i = q.size() - 1; i < q.size(); i++) {
                TrieNode *now = q[i];
                for (auto it : now->next) {
                    q.push_back(it.second);
                }
            }
        }
        for (TrieNode * node : q)
            delete node;
        q.clear();
        root = new TrieNode();
        specialRoot = nullptr;
        tokenToStringDict.clear();
        tokenToScoreDict.clear();
        stringToTokenDict.clear();
    }

    void Tokenizer::Insert(const std::string &s, int tokenId, float score) {
        TrieNode *now = this->root;
        for (int i = 0; i < s.size(); i++) {
            if (now->next.find(s[i]) == now->next.end()) {
                now->next[s[i]] = new TrieNode();
            }
            now = now->next[s[i]];
        }
        now->tokenId = tokenId;
        now->score = score;
        tokenToStringDict[tokenId] = s;
        tokenToScoreDict[tokenId] = score;
        stringToTokenDict[s] = tokenId;
    }

    void Tokenizer::SetSpecialTokens(const std::map<std::string, int>& specialTokenMap) {
        if (specialRoot == nullptr)
            specialRoot = new TrieNode();
        for (auto &it : specialTokenMap) {
            TrieNode *now = this->specialRoot;
            for (int i = 0; i < it.first.size(); i++) {
                if (now->next.find(it.first[i]) == now->next.end()) {
                    now->next[it.first[i]] = new TrieNode();
                }
                now = now->next[it.first[i]];
            }
            now->tokenId = it.second;
            now->score = 0.0f;
            tokenToStringDict[it.second] = it.first;
            stringToTokenDict[it.first] = it.second;
            specialTokens.push_back(it.first);
        }
    }

    void Tokenizer::TryMergePairs(std::vector<Symbol> &symbols, int l, int r, std::priority_queue <SymbolPairs> &q) {
        if (l == -1 || r == -1 || symbols[l].len == 0 || symbols[r].len == 0) {
            return;
        }
        auto now = symbols[l].node;
        char *s = symbols[r].s;
        int pos = symbols[r].pos, len = symbols[r].len;
        for (int i = pos; i < pos + len; i++) {
            if (now->next.find(s[i]) != now->next.end()) {
                now = now->next[s[i]];
            } else {
                return;
            }
        }
        if (now->tokenId == -999999) {
            return;
        }
        q.push(SymbolPairs(now->score, l, r, symbols[l].len + symbols[r].len));
    }

    int Tokenizer::GetRank(std::vector<Symbol> &symbols,  std::vector<std::pair<int, int>> &partitions, int idx, int skip) {
        if (idx + skip + 2 >= partitions.size()) {
            return std::numeric_limits<int>::max();
        }
        auto s = symbols[0].s + symbols[0].pos;
        std::string key(s + partitions[idx].first, s + partitions[idx + skip + 2].first);
        if (stringToTokenDict.find(key) != stringToTokenDict.end()) {
            return stringToTokenDict[key];
        }
        return std::numeric_limits<int>::max();
    }

    std::string Tokenizer::Normalize(const std::string &ori) {
        if (this->byteAsChar) {
            std::wstring ws(ori.size(), L' ');
            for (int i=0; i < ori.length(); i++) {
                wchar_t wi = static_cast<wchar_t>(static_cast<unsigned char>(ori[i]));
                if (charByteDict.find(wi) != charByteDict.end()) {
                    wi = charByteDict[wi];
                }
                ws[i] = wi;
            }
            return converter.to_bytes(ws);  
        }
        std::string blank = "";
        blank += 226, blank += 150, blank += 129;
        std::string s = this->addDummyPrefix ? blank : "";
        if (15 < ori.size() && ori.substr(0, 15) == "<FLM_FIX_TOKEN_") {
            s = "";
        }
        for (int i = 0; i < ori.size(); i++) {
            if (ori[i] == ' ') {
                if (!(this->removeExtraWhitespaces && i > 0 && ori[i - 1] == ' ')) {
                    s += blank;
                }
            } else {
                s += ori[i];
            }
        }
        return s;
    }

    std::vector<float> Tokenizer::Encode(const std::string &ori) {
        if (this->type == TokenizerType::BPE) {} 
        else if (this->type == TokenizerType::GLM) 
        {
            const std::map<std::string, int> specialTokens = {{"[MASK]", 50003}, {"[sMASK]", 50008}, {"[gMASK]", 50009}};
            std::string s = Normalize(ori);
            std::vector<float> v;
            int findPos = 0;
            while (findPos < s.length()) {
                int nextSpecialToken = -1;
                int nextSpecialTokenPos = -1;
                int nextSpecialTokenLen = -1;
                //查看是输入的字符串是否有special token
                for (auto p : specialTokens) {
                    int ind = s.find(p.first, findPos);
                    if (ind >= 0 && (nextSpecialTokenPos < 0 || ind < nextSpecialTokenPos)) {
                        nextSpecialTokenPos = ind;
                        nextSpecialToken = p.second;
                        nextSpecialTokenLen = p.first.length();
                    }
                }
                std::string subStr;
                if (nextSpecialTokenPos < 0) {
                    subStr = s.substr(findPos);
                    findPos = s.length();
                } else {
                    subStr = s.substr(findPos, nextSpecialTokenPos - findPos);
                    findPos = nextSpecialTokenPos + nextSpecialTokenLen;
                }
                if (subStr.length() > 0) {
#ifdef USE_SENTENCEPIECE
                    if (spProcessor!=nullptr) {
                        std::vector<int> ids;
                        spProcessor->Encode(subStr, &ids);
                        fo r(int id : ids) {
                            v.push_back(id);
                        }
                    } else {
#endif
                    std::vector<Symbol> symbols;
                    for (int i = 0; i < subStr.size(); i++) {
                        int tokenId = -999999, pos = i - 1;
                        TrieNode *now = this->root;
                        for (int j = i; j < subStr.size(); j++) {
                            if (now->next.find(subStr[j]) != now->next.end()) {
                                now = now->next[subStr[j]];
                                if (now->tokenId != -999999) {
                                    tokenId = now->tokenId;
                                    pos = j;
                                    break;
                                }
                            } else {
                                break;
                            }
                        }
                        if (pos >= i) {
                            symbols.push_back(Symbol(now, (char *) subStr.data(), i, pos - i + 1, (int) symbols.size() - 1,
                                                     (int) symbols.size() + 1, -999999));
                            i = pos;
                        } else {
                            symbols.push_back(Symbol(nullptr, (char *) subStr.data(), i, 0, (int) symbols.size() - 1,
                                                     (int) symbols.size() + 1, -999999));
                        }
                    }
                    symbols.back().next = -1;

                    std::priority_queue<SymbolPairs> workQueue;
                    for (int i = 1; i < symbols.size(); i++) {
                        TryMergePairs(symbols, i - 1, i, workQueue);
                    }

                    while (!workQueue.empty()) {
                        auto top = workQueue.top();
                        workQueue.pop();
                        if (symbols[top.l].len == 0 || symbols[top.r].len == 0 ||
                                symbols[top.l].len + symbols[top.r].len != top.size) {
                            continue;
                        }

                        for (int i = symbols[top.r].pos; i < symbols[top.r].pos + symbols[top.r].len; i++) {
                            symbols[top.l].node = symbols[top.l].node->next[symbols[top.r].s[i]];
                        }
                        symbols[top.l].len += symbols[top.r].len;
                        symbols[top.r].len = 0;
                        symbols[top.l].next = symbols[top.r].next;
                        if (symbols[top.r].next >= 0) {
                            symbols[symbols[top.r].next].prev = top.l;
                        }

                        TryMergePairs(symbols, symbols[top.l].prev, top.l, workQueue);
                        TryMergePairs(symbols, top.l, symbols[top.l].next, workQueue);
                    }
                    for (int i = 0; i < symbols.size(); i++) {
                        if (symbols[i].len > 0) {
                            v.push_back(symbols[i].node->tokenId);
                        } else if (symbols[i].node == nullptr) {
                            if (symbols[i].fixId != -999999) {
                                v.push_back(symbols[i].fixId);
                            } else {
                                // 未识别的字符
                                uint8_t c = (uint8_t) (symbols[i].s[symbols[i].pos]);
                                std::string now = "<0x00>";
                                now[3] = (c / 16 > 9 ? ('A' + c / 16 - 10) : ('0' + c / 16));
                                now[4] = (c % 16 > 9 ? ('A' + c % 16 - 10) : ('0' + c % 16));
                                if (stringToTokenDict.find(now) != stringToTokenDict.end()) {
                                    v.push_back(stringToTokenDict[now]);
                                }
                            }
                        }
                    }
#ifdef USE_SENTENCEPIECE
                    }
#endif
                }
                if (nextSpecialTokenPos >= 0) {
                    v.push_back(nextSpecialToken);
                }
            }
            return v;
            //return Data (DataType::FLOAT32, {1, (int)v.size()}, v);
        } 
        else if (this->type == TokenizerType::QWEN) 
        {
            
        } else 
        {
    
        }
    }




    std::string Tokenizer::Decode(const std::vector<int> &tokens) {
        std::string ret = "";
        for (int i = 0; i < tokens.size(); i++) {
            std::string s = tokenToStringDict[tokens[i]];
            if (s.size() == 6 && s.substr(0, 3) == "<0x" && s.back() == '>') {
                int c = 0;
                for (int i = 3; i < 5; i++) {
                    c *= 16;
                    if (s[i] >= '0' && s[i] <= '9') {
                        c += (s[i] - '0');
                    } else {
                        c += (s[i] - 'A' + 10);
                    }
                }

                s = " ";
                s[0] = c;
            }
            if (s == "<n>") {
                ret += "\n";
            } else if (s == "<|tab|>") {
                ret += "\t";
            } else {
                ret += s;
            }
        }

        std::string blank = "";
        blank += 226, blank += 150, blank += 129;
        while (true) {
            std::string::size_type pos(0);
            if ((pos = ret.find(blank)) != std::string::npos)
                ret.replace(pos, blank.length(), " ");
            else break;
        }
        if (this->byteAsChar) {
            std::wstring wret = converter.from_bytes(ret);
            std::string decoded(wret.size(), ' ');
            for (int i=0; i < wret.length(); i++) {
                if (byteCharDict.find(wret[i]) != byteCharDict.end()) {
                    wret[i] = byteCharDict[wret[i]];
                }
                decoded[i] = static_cast<char>(wret[i]);
            }
            ret = decoded;
        }
        int pos = ret.find("<|blank_");
        if (pos != -1) {
            int space_num = atoi(ret.substr(8, ret.size() - 10).c_str());
            return std::string(space_num, ' ');
        }

        return ret;
    }

