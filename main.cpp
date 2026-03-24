#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>
#include <ext/pb_ds/assoc_container.hpp>

using namespace std;
using namespace __gnu_pbds;

const uint32_t MOD = 1e9 + 7;
using state_t = uint64_t; 
const int BITS = 4;
const state_t MASK = (1ULL << BITS) - 1;

struct custom_hash {
    static uint64_t splitmix64(uint64_t x) {
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }
    size_t operator()(state_t x) const {
        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64((uint64_t)x + FIXED_RANDOM);
    }
};

long long power(long long base, long long exp) {
    long long res = 1;
    base %= MOD;
    while (exp > 0) {
        if (exp & 1) res = (res * base) % MOD;
        base = (base * base) % MOD;
        exp >>= 1;
    }
    return res;
}

long long modInverse(long long n) { return power(n, MOD - 2); }

vector<long long> berlekamp_massey(const vector<long long>& s) {
    vector<long long> C = {1}, B = {1};
    int L = 0, m = 1;
    long long b = 1;

    for (size_t i = 0; i < s.size(); ++i) {
        unsigned long long d_fast = 0;
        for (int j = 0; j <= L; ++j) {
            d_fast += (unsigned long long)C[j] * s[i - j];
            if (d_fast >= 1700000000000000000ULL) {
                d_fast %= MOD;
            }
        }
        long long d = d_fast % MOD;

        if (d == 0) {
            m++;
        } else {
            vector<long long> T = C;
            long long c_val = (d * modInverse(b)) % MOD;
            while (C.size() <= B.size() + m) C.push_back(0);
            for (size_t j = 0; j < B.size(); ++j) {
                C[j + m] = (C[j + m] - c_val * B[j]) % MOD;
                if (C[j + m] < 0) C[j + m] += MOD;
            }
            if (2 * L <= (int)i) { L = i + 1 - L; B = T; b = d; m = 1; } 
            else { m++; }
        }
    }
    C.resize(L + 1);
    return C;
}

inline state_t get_canonical(state_t state, int m) {
    int mapping[16];
    for (int i = 0; i < 16; ++i) mapping[i] = -1;
    int next_color = 0;
    state_t canonical = 0;
    for (int i = m - 1; i >= 0; --i) {
        int c = (state >> (i * BITS)) & MASK;
        if (mapping[c] == -1) mapping[c] = next_color++;
        canonical = (canonical << BITS) | mapping[c];
    }
    return canonical;
}

void init_dfs_layer0(int idx, int last_c, int max_c, state_t curr, int m, int q, 
                     gp_hash_table<state_t, int, custom_hash>& canonical_to_id, 
                     vector<state_t>& unique_states,
                     vector<uint32_t>& v_init) {
    if (idx == m) {
        state_t cid = get_canonical(curr, m);

        uint64_t weight = 1;
        for (int i = 0; i <= max_c; ++i) {
            weight = (weight * (q - i)) % MOD;
        }
        
        auto it = canonical_to_id.find(cid);
        if (it == canonical_to_id.end()) {
            canonical_to_id[cid] = unique_states.size();
            unique_states.push_back(cid);
            v_init.push_back(weight);
        } else {
            v_init[it->second] = (v_init[it->second] + weight) % MOD;
        }
        return;
    }
    
    int limit = (idx == 0) ? 0 : min(q - 1, max_c + 1);
    
    for (int c = 0; c <= limit; ++c) {
        if (idx > 0 && c == last_c) continue;
        state_t next_curr = curr | ((state_t)c << ((m - 1 - idx) * BITS));
        init_dfs_layer0(idx + 1, c, max(max_c, c), next_curr, m, q, canonical_to_id, unique_states, v_init);
    }
}

void solve(int m, int q, int N_TERMS) {
    auto start_time = chrono::high_resolution_clock::now();

    vector<state_t> layer_states[20];
    gp_hash_table<state_t, int, custom_hash> layer_map[20];

    vector<vector<pair<uint32_t, uint32_t>>> pull_adj[20]; 
    
    vector<uint32_t> v_init;
    init_dfs_layer0(0, -1, 0, 0, m, q, layer_map[0], layer_states[0], v_init);

    for (int row = 0; row < m; ++row) {
        int target_layer = (row == m - 1) ? 0 : row + 1;
        
        for (int i = 0; i < layer_states[row].size(); ++i) {
            state_t state = layer_states[row][i]; 

            int max_color = -1;
            for(int j = 0; j < m; ++j) {
                int c = (state >> (j * BITS)) & MASK;
                if(c > max_color) max_color = c;
            }
            int k = max_color + 1; 

            int c_old = (state >> ((m - 1 - row) * BITS)) & MASK;
            int c_up = (row == 0) ? -1 : ((state >> ((m - 1 - (row - 1)) * BITS)) & MASK);

            for (int c_new = 0; c_new <= k; ++c_new) {
                if (c_new == c_old || c_new == c_up) continue;
                if (c_new == k && k >= q) continue;
                
                state_t next_state = state & ~((state_t)MASK << ((m - 1 - row) * BITS)); 
                next_state |= ((state_t)c_new << ((m - 1 - row) * BITS)); 
                state_t can_next = get_canonical(next_state, m); 

                uint32_t edge_weight = (c_new == k) ? (q - k) % MOD : 1;

                int next_id;
                auto it = layer_map[target_layer].find(can_next);
                if (it == layer_map[target_layer].end()) {
                    next_id = layer_states[target_layer].size();
                    layer_map[target_layer][can_next] = next_id;
                    layer_states[target_layer].push_back(can_next);
                    pull_adj[row].push_back(vector<pair<uint32_t, uint32_t>>()); 
                } else {
                    next_id = it->second;
                }
                
                while(pull_adj[row].size() <= next_id) {
                    pull_adj[row].push_back(vector<pair<uint32_t, uint32_t>>());
                }
                
                pull_adj[row][next_id].push_back({i, edge_weight});
            }
        }
    }

    vector<uint32_t> csr_row_ptr[20];
    vector<uint32_t> csr_col_idx[20];
    vector<uint32_t> csr_weight[20];
    
    for (int row = 0; row < m; ++row) {
        int next_size = layer_states[row == m - 1 ? 0 : row + 1].size();
        csr_row_ptr[row].assign(next_size + 1, 0);
        
        int total_edges = 0;
        for (int i = 0; i < next_size; ++i) {
            if (i < pull_adj[row].size()) total_edges += pull_adj[row][i].size();
        }
        csr_col_idx[row].reserve(total_edges);
        csr_weight[row].reserve(total_edges);
        
        for (int i = 0; i < next_size; ++i) {
            csr_row_ptr[row][i] = csr_col_idx[row].size();
            if (i < pull_adj[row].size()) {
                for (const auto& edge : pull_adj[row][i]) {
                    csr_col_idx[row].push_back(edge.first);
                    csr_weight[row].push_back(edge.second);
                }
            }
        }
        csr_row_ptr[row][next_size] = csr_col_idx[row].size();
        
        vector<vector<pair<uint32_t, uint32_t>>>().swap(pull_adj[row]);
    }

    int num_initial_states = layer_states[0].size();
    
    mt19937 rng(42);
    uniform_int_distribution<uint32_t> dist(1, MOD - 1);
    vector<uint32_t> random_weights(num_initial_states);
    for (int i = 0; i < num_initial_states; ++i) random_weights[i] = dist(rng);

    vector<long long> sequence;
    sequence.reserve(N_TERMS);
    
    vector<vector<uint32_t>> v_layers(m + 1);
    for (int row = 0; row <= m; ++row) {
        v_layers[row].resize(layer_states[row == m ? 0 : row].size(), 0);
    }
    
    for (int i = 0; i < num_initial_states; ++i) v_layers[0][i] = v_init[i];
    
    for (int n = 0; n < N_TERMS; ++n) {
        uint64_t current_sum = 0;
        const uint32_t* v_layer_0 = v_layers[0].data();
        const uint32_t* weights = random_weights.data();
        
        for (int i = 0; i < num_initial_states; ++i) {
            current_sum = (current_sum + (uint64_t)v_layer_0[i] * weights[i]) % MOD;
        }
        sequence.push_back(current_sum);

        for (int row = 0; row < m; ++row) {
            int next_size = layer_states[row == m - 1 ? 0 : row + 1].size();
            int target_layer = row + 1;
            
            const uint32_t* row_ptr = csr_row_ptr[row].data();
            const uint32_t* col_idx = csr_col_idx[row].data();
            const uint32_t* edge_weights = csr_weight[row].data();
            const uint32_t* v_in = v_layers[row].data();
            uint32_t* v_out = v_layers[target_layer].data();

            #pragma omp parallel for schedule(static, 256)
            for (int i = 0; i < next_size; ++i) {
                uint64_t sum = 0; 
                uint32_t start = row_ptr[i];
                uint32_t end = row_ptr[i + 1];
                
                for (uint32_t p = start; p < end; ++p) {
                    sum += (uint64_t)v_in[col_idx[p]] * edge_weights[p];
                    if (sum >= 1700000000000000000ULL) sum %= MOD; 
                }
                
                v_out[i] = sum % MOD;
            }
        }
        v_layers[0].swap(v_layers[m]); 
    }

    vector<long long> C = berlekamp_massey(sequence);
    
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end_time - start_time;

    cout << C.size() - 1 << " ";
}

int main() {
 
    int terms = 2*3000 + 50; 
    cout << "q = 5 : ";
    for (int m=1; m<=12; ++m) solve(m, 5, terms); 
    cout << endl;
    cout << "q = 6 : ";
    for (int m=1; m<=12; ++m) solve(m, 6, terms); 
    cout << endl;
    cout << "q = 7 : ";
    for (int m=1; m<=12; ++m) solve(m, 7, terms); 
    cout << endl;
    cout << "q = 8 : ";
    for (int m=1; m<=11; ++m) solve(m, 8, terms); 
    cout << endl;
    cout << "q = 9 : ";
    for (int m=1; m<=10; ++m) solve(m, 9, terms); 
    cout << endl;
    cout << "q = 10 : ";
    for (int m=1; m<=9; ++m) solve(m, 10, terms); 
    cout << endl;

    
    return 0;
}
