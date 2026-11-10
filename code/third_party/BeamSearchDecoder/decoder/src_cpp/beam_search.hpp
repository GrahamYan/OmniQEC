// This file is part of BeamSearchDecoder.

// Copyright (c) 2025 IonQ, Inc., all rights reserved

// Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
// 4.0 International License (CC BY-NC-SA 4.0).

// You may obtain a copy of the License at:
// https://creativecommons.org/licenses/by-nc-sa/4.0/


#ifndef BP_H
#define BP_H

#include <utility>
#include <algorithm>
#include <iostream>
#include <vector>
#include <queue>
#include <unordered_set>
#include <memory>
#include <iterator>
#include <cmath>
#include <limits>
#include <random>
#include <chrono>
#include <stdexcept> // required for std::runtime_error
#include <set>

#include "math.h"
#include "sparse_matrix_base.hpp"
#include "gf2sparse.hpp"

namespace ldpc {
    namespace bp {
        class BpEntry : public ldpc::sparse_matrix_base::EntryBase<BpEntry> {
        public:
            double bit_to_check_msg = 0.0;
            double check_to_bit_msg = 0.0;

            ~BpEntry() = default;
        };
        using BpSparse = ldpc::gf2sparse::GF2Sparse<BpEntry>;

        class BeamSearchDecoder {
            // TODO properties should be private and only accessible via getters and setters
        public:
            BpSparse &pcm;
            std::vector<double> channel_probabilities;
            int check_count;
            int bit_count;
            int max_rounds;
            int beam_width;
            int num_results;
            int initial_iters;
            int iters_per_round;
            std::vector<uint8_t> decoding;
            std::vector<uint8_t> candidate_syndrome;

            std::vector<double> log_prob_ratios;
            std::vector<double> initial_log_prob_ratios;
            int iterations;
            bool converge;

            BeamSearchDecoder(
                    BpSparse &parity_check_matrix,
                    std::vector<double> channel_probabilities,
                    int max_rounds = 10,
                    int beam_width = 8,
                    int num_results = 1,
                    int initial_iters = 30,
                    int iters_per_round = 20) :
                    pcm(parity_check_matrix), channel_probabilities(std::move(channel_probabilities)),
                    check_count(pcm.m), bit_count(pcm.n), max_rounds(max_rounds), beam_width(beam_width), num_results(num_results),
                    initial_iters(initial_iters), iters_per_round(iters_per_round),
                    iterations(0) //the parity check matrix is passed in by reference
            {

                this->initial_log_prob_ratios.resize(bit_count);
                this->log_prob_ratios.resize(bit_count);
                this->candidate_syndrome.resize(check_count);
                this->decoding.resize(bit_count);
                this->converge = 0;


                if (this->channel_probabilities.size() != this->bit_count) {
                    throw std::runtime_error(
                            "Channel probabilities vector must have length equal to the number of bits");
                }
            }

            ~BeamSearchDecoder() = default;

            void initialise_log_domain_bp() {
                // initialise BP
                for (int i = 0; i < this->bit_count; i++) {
                    this->initial_log_prob_ratios[i] = std::log(
                            (1 - this->channel_probabilities[i]) / this->channel_probabilities[i]);

                    for (auto &e: this->pcm.iterate_column(i)) {
                        e.bit_to_check_msg = this->initial_log_prob_ratios[i];
                    }
                }
            }

            std::vector<uint8_t> &decode(std::vector<uint8_t> &syndrome) {

                this->converge = 0;

                this->initialise_log_domain_bp();

                using Pair = std::pair<double, int>;
                std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> min_pq;
                std::vector<int> bit_masks(this->bit_count, -1);
                std::vector<double> LLR_sums(this->bit_count, 0);
                int edge_msgs_length = 0;
                for (int i = 0; i < this->bit_count; i++) {
                    edge_msgs_length += this->pcm.iterate_column(i).entry_count;
                }
                std::vector<std::vector<double>> edge_msgs(4 * this->beam_width, std::vector<double>(edge_msgs_length, 0));
                std::vector<std::vector<int>> fixed_indices(4 * this->beam_width, std::vector<int>(this->max_rounds + 1, 0));
                std::vector<std::vector<int>> fixed_values(4 * this->beam_width, std::vector<int>(this->max_rounds + 1, 0));
                std::vector<int> converged_value(4 * this->beam_width, -1);
                std::vector<int> explore_list;
                std::vector<uint8_t> cur_decoding(this->bit_count, 0);
                int start = 0, next_start = 2 * this->beam_width;
                double min_dec_weight = std::numeric_limits<double>::max();
                double cur_dec_weight;
                int sol_so_far = 0;

                // initial bp iterations
                for (int it = 1; it <= this->initial_iters; it++) {
                    //check to bit updates
                    for (int i = 0; i < check_count; i++) {

                        this->candidate_syndrome[i] = 0;
                        int total_sgn = 0;
                        int sgn = 0;
                        total_sgn = syndrome[i];
                        double temp = std::numeric_limits<double>::max();

                        for (auto &e: this->pcm.iterate_row(i)) {
                            if (e.bit_to_check_msg <= 0) {
                                total_sgn += 1;
                            }
                            e.check_to_bit_msg = temp;
                            double abs_bit_to_check_msg = std::abs(e.bit_to_check_msg);
                            if (abs_bit_to_check_msg < temp) {
                                temp = abs_bit_to_check_msg;
                            }
                        }

                        temp = std::numeric_limits<double>::max();
                        for (auto &e: this->pcm.reverse_iterate_row(i)) {
                            sgn = total_sgn;
                            if (e.bit_to_check_msg <= 0) {
                                sgn += 1;
                            }
                            if (temp < e.check_to_bit_msg) {
                                e.check_to_bit_msg = temp;
                            }

                            int message_sign = (sgn % 2 == 0) ? 1.0 : -1.0;

                            e.check_to_bit_msg *= message_sign;

                            double abs_bit_to_check_msg = std::abs(e.bit_to_check_msg);
                            if (abs_bit_to_check_msg < temp) {
                                temp = abs_bit_to_check_msg;
                            }
                        }
                    }

                    //compute log probability ratios
                    for (int i = 0; i < this->bit_count; i++) {
                        double temp = initial_log_prob_ratios[i];
                        for (auto &e: this->pcm.iterate_column(i)) {
                            e.bit_to_check_msg = temp;
                            temp += e.check_to_bit_msg;
                        }

                        //make hard decision on basis of log probability ratio for bit i
                        this->log_prob_ratios[i] = temp;
                        if (temp <= 0) {
                            this->decoding[i] = 1;
                            for (auto &e: this->pcm.iterate_column(i)) {
                                this->candidate_syndrome[e.row_index] ^= 1;
                            }
                        } else {
                            this->decoding[i] = 0;
                        }
                    }

                    //compute bit to check update
                    for (int i = 0; i < bit_count; i++) {
                        double temp = 0;
                        for (auto &e: this->pcm.reverse_iterate_column(i)) {
                            e.bit_to_check_msg += temp;
                            temp += e.check_to_bit_msg;
                        }
                    }

                    //calculate statistics
                    for (int i = 0; i < this->bit_count; i++) {
                        LLR_sums[i] += this->log_prob_ratios[i];
                    }

                    if (std::equal(candidate_syndrome.begin(), candidate_syndrome.end(), syndrome.begin())) {
                        this->converge = true;
                    }

                    this->iterations = it;

                    if (this->converge) {
                        min_dec_weight = 0;
                        for (int i = 0; i < this->bit_count; i++) {
                            if (this->decoding[i]) {
                                min_dec_weight += this->initial_log_prob_ratios[i];
                            }
                        }
                        sol_so_far++;
                        if (sol_so_far == this->num_results) {
                            return this->decoding;
                        }
                        break;
                    }
                }

                //Initialize for subsequent rounds
                double min_score = std::abs(LLR_sums[0]);
                int min_idx = 0;
                for (int i = 0; i < this->bit_count; i++) {
                    // skip qubits whose Tanner graph degree is <=2
                    if (this->pcm.iterate_column(i).entry_count <= 2) continue;
                    if (min_score > std::abs(LLR_sums[i])) {
                        min_score = std::abs(LLR_sums[i]);
                        min_idx = i;
                    }
                }
                fixed_indices[0][0] = min_idx;
                if (this->converge) {
                    converged_value[0] = cur_decoding[min_idx];
                } else {
                    converged_value[0] = -1;
                }
                int msg_idx = 0;
                for (int i = 0; i < this->bit_count; i++) {
                    for (auto &e: this->pcm.iterate_column(i)) {
                        edge_msgs[0][msg_idx] = e.bit_to_check_msg;
                        msg_idx++;
                    }
                }
                explore_list.push_back(0);

                // subsequent rounds
                for (int round = 0; round < this->max_rounds; round++) {
                    int store_idx = 0;

                    for (int path_id = 0; path_id < 2 * explore_list.size(); path_id++) {
                        int list_ele = explore_list[path_id / 2];
                        int bit_val = path_id % 2;
                        if (converged_value[start + list_ele] == bit_val) continue;
                        fixed_values[start + list_ele][round] = bit_val;
                        for (int i = 0; i <= round; i++) {
                            bit_masks[fixed_indices[start + list_ele][i]] = fixed_values[start + list_ele][i];
                            if (fixed_values[start + list_ele][i]) {
                                for (auto &e: this->pcm.iterate_column(fixed_indices[start + list_ele][i])) {
                                    syndrome[e.row_index] ^= 1;
                                }
                            }
                        }
                        // Initialize bit_to_check_msg from the best paths in the previous round
                        msg_idx = 0;
                        for (int i = 0; i < this->bit_count; i++) {
                            if (bit_masks[i] != -1) {
                                msg_idx += this->pcm.iterate_column(i).entry_count;
                                continue;
                            }
                            for (auto &e: this->pcm.iterate_column(i)) {
                                e.bit_to_check_msg = edge_msgs[start + list_ele][msg_idx];
                                msg_idx++;
                            }
                        }

                        this->converge = 0;
                        for (int i = 0; i < this->bit_count; i++) LLR_sums[i] = 0;

                        //main interation loop
                        for (int it = 1; it <= this->iters_per_round; it++) {
                            //check to bit updates
                            for (int i = 0; i < check_count; i++) {

                                this->candidate_syndrome[i] = 0;
                                int total_sgn = 0;
                                int sgn = 0;
                                total_sgn = syndrome[i];
                                double temp = std::numeric_limits<double>::max();

                                for (auto &e: this->pcm.iterate_row(i)) {
                                    // ignore fixed bits
                                    if (bit_masks[e.col_index] != -1) continue;
                                    if (e.bit_to_check_msg <= 0) {
                                        total_sgn += 1;
                                    }
                                    e.check_to_bit_msg = temp;
                                    double abs_bit_to_check_msg = std::abs(e.bit_to_check_msg);
                                    if (abs_bit_to_check_msg < temp) {
                                        temp = abs_bit_to_check_msg;
                                    }
                                }

                                temp = std::numeric_limits<double>::max();
                                for (auto &e: this->pcm.reverse_iterate_row(i)) {
                                    // ignore fixed bits
                                    if (bit_masks[e.col_index] != -1) continue;
                                    sgn = total_sgn;
                                    if (e.bit_to_check_msg <= 0) {
                                        sgn += 1;
                                    }
                                    if (temp < e.check_to_bit_msg) {
                                        e.check_to_bit_msg = temp;
                                    }

                                    int message_sign = (sgn % 2 == 0) ? 1.0 : -1.0;

                                    e.check_to_bit_msg *= message_sign;

                                    double abs_bit_to_check_msg = std::abs(e.bit_to_check_msg);
                                    if (abs_bit_to_check_msg < temp) {
                                        temp = abs_bit_to_check_msg;
                                    }
                                }
                            }

                            //compute log probability ratios
                            for (int i = 0; i < this->bit_count; i++) {
                                if (bit_masks[i] != -1) continue;
                                double temp = initial_log_prob_ratios[i];
                                for (auto &e: this->pcm.iterate_column(i)) {
                                    e.bit_to_check_msg = temp;
                                    temp += e.check_to_bit_msg;
                                }

                                //make hard decision on basis of log probability ratio for bit i
                                this->log_prob_ratios[i] = temp;
                                if (temp <= 0) {
                                    cur_decoding[i] = 1;
                                    for (auto &e: this->pcm.iterate_column(i)) {
                                        this->candidate_syndrome[e.row_index] ^= 1;
                                    }
                                } else {
                                    cur_decoding[i] = 0;
                                }
                            }

                            //compute bit to check update
                            for (int i = 0; i < bit_count; i++) {
                                if (bit_masks[i] != -1) continue;
                                double temp = 0;
                                for (auto &e: this->pcm.reverse_iterate_column(i)) {
                                    e.bit_to_check_msg += temp;
                                    temp += e.check_to_bit_msg;
                                }
                            }

                            //calculate statistics
                            for (int i = 0; i < this->bit_count; i++) {
                                if (bit_masks[i] == -1) {
                                    LLR_sums[i] += this->log_prob_ratios[i];
                                }
                            }

                            if (std::equal(candidate_syndrome.begin(), candidate_syndrome.end(), syndrome.begin())) {
                                this->converge = true;
                            }

                            this->iterations = it;

                            if (this->converge) {
                                cur_dec_weight = 0;
                                for (int i = 0; i < this->bit_count; i++) {
                                    if (bit_masks[i] != -1) {
                                        cur_decoding[i] = bit_masks[i];
                                    }
                                    if (cur_decoding[i]) {
                                        cur_dec_weight += this->initial_log_prob_ratios[i];
                                    }
                                }
                                if (cur_dec_weight < min_dec_weight) {
                                    min_dec_weight = cur_dec_weight;
                                    this->decoding = cur_decoding;
                                }
                                sol_so_far++;
                                if (sol_so_far == this->num_results) {
                                    return this->decoding;
                                }
                                break;
                            }
                        }

                        // Compare different paths.
                        double score = 0;
                        for (int i = 0; i < this->bit_count; i++) {
                            if (bit_masks[i] == -1) {
                                score += std::abs(LLR_sums[i]);
                            }
                        }
                        score = score / static_cast<double>(this->iterations);
                        if (min_pq.size() >= this->beam_width && score < min_pq.top().first) {
                            // restore the bit_masks vector and the syndrome vector
                            for (int i = 0; i <= round; i++) {
                                bit_masks[fixed_indices[start + list_ele][i]] = -1;
                                if (fixed_values[start + list_ele][i]) {
                                    for (auto &e: this->pcm.iterate_column(fixed_indices[start + list_ele][i])) {
                                        syndrome[e.row_index] ^= 1;
                                    }
                                }
                            }
                            continue;
                        }
                        for (int i = 0; i <= round; i++) {
                            fixed_indices[next_start + store_idx][i] = fixed_indices[start + list_ele][i];
                            fixed_values[next_start + store_idx][i] = fixed_values[start + list_ele][i];
                        }
                        min_score = std::numeric_limits<double>::max();
                        for (int i = 0; i < this->bit_count; i++) {
                            if (bit_masks[i] != -1) continue;
                            // skip qubits whose Tanner graph degree is <=2
                            if (this->pcm.iterate_column(i).entry_count <= 2) continue;
                            if (min_score > std::abs(LLR_sums[i])) {
                                min_score = std::abs(LLR_sums[i]);
                                min_idx = i;
                            }
                        }
                        fixed_indices[next_start + store_idx][round + 1] = min_idx;
                        if (this->converge) {
                            converged_value[next_start + store_idx] = cur_decoding[min_idx];
                        } else {
                            converged_value[next_start + store_idx] = -1;
                        }

                        msg_idx = 0;
                        for (int i = 0; i < this->bit_count; i++) {
                            if (bit_masks[i] != -1) {
                                msg_idx += this->pcm.iterate_column(i).entry_count;
                                continue;
                            }
                            for (auto &e: this->pcm.iterate_column(i)) {
                                edge_msgs[next_start + store_idx][msg_idx] = e.bit_to_check_msg;
                                msg_idx++;
                            }
                        }

                        min_pq.push({score, store_idx});
                        if (min_pq.size() > this->beam_width) min_pq.pop();
                        store_idx++;

                        // restore the bit_masks vector and the syndrome vector
                        for (int i = 0; i <= round; i++) {
                            bit_masks[fixed_indices[start + list_ele][i]] = -1;
                            if (fixed_values[start + list_ele][i]) {
                                for (auto &e: this->pcm.iterate_column(fixed_indices[start + list_ele][i])) {
                                    syndrome[e.row_index] ^= 1;
                                }
                            }
                        }
                    }

                    // Update for next rounds. Take reverse order because we want to explore elements with higher scores first.
                    explore_list.resize(min_pq.size());
                    for (int i = explore_list.size() - 1; i >= 0; i--) {
                        explore_list[i] = min_pq.top().second;
                        min_pq.pop();
                    }

                    if (start == 0) {
                        start = 2 * this->beam_width;
                        next_start = 0;
                    } else {
                        start = 0;
                        next_start = 2 * this->beam_width;
                    }
                }
                return this->decoding;
            }
        };
    }
}  // namespace ldpc::bp

#endif
