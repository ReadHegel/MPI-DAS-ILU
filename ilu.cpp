// Optymalizacje do napisania:
// - wysyłanie tylko wyrazy po prawej stronie przekątnej
//

#include "ilu.h"

#include <mpi.h>

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

struct CSRMatrix {
    int num_rows;
    int num_cols;
    int nnz;
    std::vector<int> row_ptr;  // Size: num_rows + 1
    std::vector<int> col_idx;  // Size: nnz
    std::vector<double> val;   // Size: nnz

    static CSRMatrix from_COO(
        int N, int nnz, const int *row, const int *col, const double *val
    ) {
        CSRMatrix mat;

        mat.num_rows = N;
        mat.num_cols = N;
        mat.nnz = nnz;
        mat.row_ptr.assign(N + 1, 0);
        mat.col_idx.resize(nnz);
        mat.val.resize(nnz);

        std::vector<int> indices(nnz);
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            if (row[a] != row[b]) {
                return row[a] < row[b];
            }
            return col[a] < col[b];
        });

        for (int i = 0; i < nnz; ++i) {
            int idx = indices[i];
            int r = row[idx];

            mat.row_ptr[r + 1]++;
            mat.col_idx[i] = col[idx];
            mat.val[i] = val[idx];
        }

        for (int i = 0; i < N; ++i) {
            mat.row_ptr[i + 1] += mat.row_ptr[i];
        }

        return mat;
    }

    void add_mult_row_to_row(
        double alpha,
        const int target_local_row,
        const double *other_val,
        const int *other_col_idx,
        const int other_nnz
    ) {
        int other_i = 0;
        int i = row_ptr[target_local_row];
        int row_end = row_ptr[target_local_row + 1];

        while (other_i < other_nnz && i < row_end) {
            int other_col = other_col_idx[other_i];
            int col = col_idx[i];

            if (other_col == col) {
                val[i] += alpha * other_val[other_i];
                other_i++;
                i++;
            }
            else if (other_col < col) {
                other_i++;
            }
            else {
                i++;
            }
        }
    }

    int nnz_in_local_row(int local_row) const {
        return row_ptr[local_row + 1] - row_ptr[local_row];
    }

    int find_idx(int local_row, int col) const {
        auto it = std::lower_bound(
            col_idx.begin() + row_ptr[local_row],
            col_idx.begin() + row_ptr[local_row + 1],
            col
        );
        if (it != col_idx.begin() + row_ptr[local_row + 1] && *it == col) {
            return std::distance(col_idx.begin(), it);
        }
        throw std::runtime_error("Diagonal element not found in local row");
    }

    int find_diag_idx(int local_row, int global_offset) const {
        return find_idx(local_row, local_row + global_offset);
    }
};

struct RowElem {
    int col;
    double val;

    static std::pair<std::vector<int>, std::vector<double>> unpack(
        const std::vector<RowElem> &row_elems
    ) {
        std::vector<int> cols;
        std::vector<double> vals;
        cols.reserve(row_elems.size());
        vals.reserve(row_elems.size());

        for (const auto &elem : row_elems) {
            cols.push_back(elem.col);
            vals.push_back(elem.val);
        }

        return {cols, vals};
    }
};

struct ILUFact {
    int N;
    int rank;
    int world_size;

    int global_offset;
    int num_rows_local;

    int num_interior;
    int num_separator;
    std::vector<bool> is_separator_ready;

    int glob_to_loc_row(int global_row) const {
        return global_row - global_offset;
    }

    int loc_to_glob_row(int local_row) const {
        return local_row + global_offset;
    }

    std::vector<int> perm;
    std::vector<int> inv_perm;

    CSRMatrix LU;  // Unified matrix for both L and U

    std::vector<int> first_col_in_separator_idx;

    std::unordered_map<int, std::vector<int>> send_to_ranks;  // row -> {ranks}

    std::vector<std::vector<RowElem>> recv_row_buffers;
    std::vector<MPI_Request> active_recv_requests;
    std::vector<int> request_to_row_idx;
    std::unordered_map<int, int>
        row_idx_to_request_idx;  // global row -> request idx in
                                 // active_recv_requests
    int count_active_requests;
};

namespace {

// =======================================
// UTILITIES
// =======================================

namespace utils {

bool is_zero(double val) {
    const double EPS = 1e-12;
    return std::abs(val) < EPS;
}

int get_first_row_of_process(int rank, int world_size, int N) {
    return rank * N / world_size;
}

int get_owner_rank(int global_row, int world_size, int N) {
    return (global_row * world_size + world_size - 1) / N;
}

void print_local_dense(const struct ILUFact *ilu) {
    MPI_Barrier(MPI_COMM_WORLD);
    for (int p = 0; p < ilu->world_size; ++p) {
        if (ilu->rank == p) {
            const CSRMatrix &mat = ilu->LU;
            std::cout << "[rank " << ilu->rank << "/" << ilu->world_size

                      << "] local rows=" << ilu->LU.num_rows
                      << " offset=" << ilu->global_offset << "\n";
            for (int i = 0; i < mat.num_rows; ++i) {
                int current_col = 0;
                for (int idx = mat.row_ptr[i]; idx < mat.row_ptr[i + 1];
                     ++idx) {
                    int target_col = mat.col_idx[idx];
                    while (current_col < target_col) {
                        std::cout << std::setw(8) << "*";
                        current_col++;
                    }
                    std::cout << std::setw(8) << std::fixed
                              << std::setprecision(3) << mat.val[idx];
                    current_col++;
                }
                while (current_col < ilu->N) {
                    std::cout << std::setw(8) << "*";
                    current_col++;
                }
                std::cout << "\n";
            }
            std::cout << std::flush;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

namespace permutation {

template <typename T>
std::vector<T> apply_permutation(
    const std::vector<T> &vec, const std::vector<int> &perm
) {
    std::vector<T> permuted(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        permuted[i] = vec[perm[i]];
    }
    return permuted;
}

std::vector<int> inverse_permutation(const std::vector<int> &perm) {
    std::vector<int> inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        inv_perm[perm[i]] = i;
    }
    return inv_perm;
}
}  // namespace permutation
}  // namespace utils

void ILU(
    CSRMatrix &LU, int global_offset, const int num_rows_from_U_to_factorize
) {
    for (int i = 1; i < num_rows_from_U_to_factorize; ++i) {
        for (int k = LU.row_ptr[i]; k < LU.row_ptr[i + 1]; ++k) {
            int row_to_subtract_loc = LU.col_idx[k] - global_offset;

            if (row_to_subtract_loc >= global_offset + i) {
                break;
            }

            int diag_kk_idx =
                LU.find_diag_idx(row_to_subtract_loc, global_offset);
            LU.val[k] = LU.val[k] / LU.val[diag_kk_idx];

            LU.add_mult_row_to_row(
                -LU.val[k],
                i,
                LU.val.data() + diag_kk_idx + 1,
                LU.col_idx.data() + diag_kk_idx + 1,
                LU.row_ptr[row_to_subtract_loc + 1] -
                    LU.row_ptr[row_to_subtract_loc] - 1
            );
        }
    }
}

void distribute_data(
    int N, int nnz, int *row, int *col, double *val, struct ILUFact *ilu
) {
    CSRMatrix &recived_matrix = ilu->LU;

    auto get_first_row_of_process = [&](int rank) {
        return utils::get_first_row_of_process(rank, ilu->world_size, N);
    };

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    ilu->global_offset = get_first_row_of_process(ilu->rank);
    ilu->num_rows_local =
        get_first_row_of_process(ilu->rank + 1) - ilu->global_offset;
    ilu->N = N;

    recived_matrix.num_rows = ilu->num_rows_local;
    recived_matrix.num_cols = N;
    recived_matrix.row_ptr.resize(recived_matrix.num_rows + 1);

    std::vector<int> col_idx_sendcounts;
    std::vector<int> col_idx_displacements;
    CSRMatrix A;

    if (ilu->rank == 0) {
        A = CSRMatrix::from_COO(N, nnz, row, col, val);
        col_idx_sendcounts.resize(ilu->world_size, 0);
        col_idx_displacements.resize(ilu->world_size, 0);

        int first_row = 0;
        int last_row = get_first_row_of_process(1) - 1;

        col_idx_displacements[0] = A.row_ptr[first_row];
        col_idx_sendcounts[0] = A.row_ptr[last_row + 1] - A.row_ptr[first_row];

        recived_matrix.nnz = col_idx_sendcounts[0];
        std::copy(
            A.row_ptr.begin() + first_row,
            A.row_ptr.begin() + last_row + 2,
            recived_matrix.row_ptr.begin()
        );

        for (int r = 1; r < ilu->world_size; ++r) {
            first_row = last_row + 1;
            last_row = get_first_row_of_process(r + 1) - 1;

            col_idx_sendcounts[r] =
                A.row_ptr[last_row + 1] - A.row_ptr[first_row];
            col_idx_displacements[r] = A.row_ptr[first_row];

            int nnz_for_r = col_idx_sendcounts[r];
            MPI_Send(&nnz_for_r, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
            MPI_Send(
                &A.row_ptr[first_row],
                last_row - first_row + 2,
                MPI_INT,
                r,
                0,
                MPI_COMM_WORLD
            );
        }
    }
    else {
        MPI_Recv(
            &recived_matrix.nnz,
            1,
            MPI_INT,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
        MPI_Recv(
            recived_matrix.row_ptr.data(),
            recived_matrix.num_rows + 1,
            MPI_INT,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
    }

    recived_matrix.col_idx.resize(recived_matrix.nnz);
    recived_matrix.val.resize(recived_matrix.nnz);

    MPI_Scatterv(
        ilu->rank == 0 ? A.col_idx.data() : nullptr,
        col_idx_sendcounts.data(),
        col_idx_displacements.data(),
        MPI_INT,
        recived_matrix.col_idx.data(),
        recived_matrix.nnz,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    MPI_Scatterv(
        ilu->rank == 0 ? A.val.data() : nullptr,
        col_idx_sendcounts.data(),
        col_idx_displacements.data(),
        MPI_DOUBLE,
        recived_matrix.val.data(),
        recived_matrix.nnz,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    int offset = recived_matrix.row_ptr[0];
    for (auto &rp : recived_matrix.row_ptr) {
        rp -= offset;
    }
}

void permute_rows(
    CSRMatrix &LU,
    const int global_offset,
    const std::vector<int> &perm,
    const std::vector<int> &inv_perm
) {
    std::vector<int> perm_row_ptr;
    std::vector<int> perm_col_idx;
    std::vector<double> perm_val;

    perm_row_ptr.reserve(LU.num_rows + 1);
    perm_col_idx.reserve(LU.nnz);
    perm_val.reserve(LU.nnz);

    perm_row_ptr.push_back(0);

    for (int row_local = 0; row_local < LU.num_rows; ++row_local) {
        int original_row = inv_perm[row_local];
        int row_start = LU.row_ptr[original_row];
        int row_end = LU.row_ptr[original_row + 1];

        std::vector<std::pair<int, double>> row_data;
        row_data.reserve(row_end - row_start);

        for (int idx = row_start; idx < row_end; ++idx) {
            int global_col = LU.col_idx[idx];
            double val = LU.val[idx];

            int permuted_col = global_col;

            if (global_col >= global_offset &&
                global_col < global_offset + LU.num_rows) {
                int local_col = global_col - global_offset;
                permuted_col = global_offset + perm[local_col];
            }

            row_data.push_back({permuted_col, val});
        }

        std::sort(
            row_data.begin(),
            row_data.end(),
            [](const std::pair<int, double> &a,
               const std::pair<int, double> &b) {
                return a.first < b.first;
            }
        );

        for (const auto &item : row_data) {
            perm_col_idx.push_back(item.first);
            perm_val.push_back(item.second);
        }

        perm_row_ptr.push_back(perm_col_idx.size());
    }

    LU.row_ptr = std::move(perm_row_ptr);
    LU.col_idx = std::move(perm_col_idx);
    LU.val = std::move(perm_val);
}

void interior_separator_partition(struct ILUFact *ilu) {
    ilu->inv_perm.resize(ilu->num_rows_local);

    std::vector<int> interor_rows;
    std::vector<int> separator_rows;

    CSRMatrix &LU = ilu->LU;

    for (int row_local = 0; row_local < LU.num_rows; ++row_local) {
        bool is_separator = false;
        for (int idx = LU.row_ptr[row_local]; idx < LU.row_ptr[row_local + 1];
             ++idx) {
            if (LU.col_idx[idx] < ilu->global_offset) {
                is_separator = true;
                ilu->first_col_in_separator_idx.push_back(idx);
                break;
            }
        }

        if (is_separator) {
            separator_rows.push_back(row_local);
        }
        else {
            interor_rows.push_back(row_local);
        }
    }

    ilu->num_interior = interor_rows.size();
    ilu->num_separator = separator_rows.size();
    interor_rows.insert(
        interor_rows.end(), separator_rows.begin(), separator_rows.end()
    );
    ilu->is_separator_ready.resize(ilu->num_separator, false);

    ilu->inv_perm = std::move(interor_rows);
    ilu->perm = utils::permutation::inverse_permutation(ilu->inv_perm);

    permute_rows(LU, ilu->global_offset, ilu->perm, ilu->inv_perm);
}

void share_dependencies(struct ILUFact *ilu) {
    // Calculate my dependencies
    std::vector<std::set<int>> needed_rows_from_rank(ilu->world_size);

    for (int row_local = ilu->num_interior; row_local < ilu->num_rows_local;
         ++row_local) {
        for (int idx = ilu->LU.row_ptr[row_local];
             idx < ilu->LU.row_ptr[row_local + 1];
             ++idx) {
            int global_col = ilu->LU.col_idx[idx];

            if (global_col < ilu->global_offset) {
                int owner_rank = utils::get_owner_rank(global_col, ilu->world_size, ilu->N);
                needed_rows_from_rank[owner_rank].insert(global_col);
            }
        }
    }

    // Share and recieve amount of rows each rank needs from other ranks
    std::vector<int> requests_count_to_send(ilu->world_size, 0);
    std::vector<int> requests_count_to_recv(ilu->world_size, 0);

    for (int p = 0; p < ilu->world_size; ++p) {
        requests_count_to_recv[p] = needed_rows_from_rank[p].size();
    }

    MPI_Alltoall(
        requests_count_to_recv.data(),
        1,
        MPI_INT,
        requests_count_to_send.data(),
        1,
        MPI_INT,
        MPI_COMM_WORLD
    );

    std::vector<std::vector<int>> rows_to_recv(ilu->world_size);
    std::vector<std::vector<int>> rows_to_send(ilu->world_size);
    std::vector<MPI_Request> requests;

    // Send information about which rows do I need
    for (int p = 0; p < ilu->world_size; ++p) {
        if (requests_count_to_recv[p] > 0) {
            rows_to_recv[p].assign(
                needed_rows_from_rank[p].begin(), needed_rows_from_rank[p].end()
            );
            MPI_Request req;
            MPI_Isend(
                rows_to_recv[p].data(),
                requests_count_to_recv[p],
                MPI_INT,
                p,
                ilu->N + 1,
                MPI_COMM_WORLD,
                &req
            );
            requests.push_back(req);
        }
    }

    // Receive information about which rows do other ranks need from me
    for (int p = 0; p < ilu->world_size; ++p) {
        if (requests_count_to_send[p] > 0) {
            rows_to_send[p].resize(requests_count_to_send[p]);
            MPI_Request req;
            MPI_Irecv(
                rows_to_send[p].data(),
                requests_count_to_send[p],
                MPI_INT,
                p,
                ilu->N + 1,
                MPI_COMM_WORLD,
                &req
            );
            requests.push_back(req);
        }
    }

    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    for (int p = 0; p < ilu->world_size; ++p) {
        for (int global_row : rows_to_send[p]) {
            ilu->send_to_ranks[global_row].push_back(p);
        }
    }

    std::vector<std::vector<int>> nnz_to_send(ilu->world_size);
    std::vector<std::vector<int>> nnz_to_recv(ilu->world_size);
    requests.clear();

    for (int p = 0; p < ilu->world_size; ++p) {
        if (!rows_to_send[p].empty()) {
            nnz_to_send[p].resize(rows_to_send[p].size());
            for (size_t i = 0; i < rows_to_send[p].size(); ++i) {
                int global_row = rows_to_send[p][i];
                int local_row = global_row - ilu->global_offset;
                nnz_to_send[p][i] = ilu->LU.nnz_in_local_row(local_row);
            }

            MPI_Request req;
            MPI_Isend(
                nnz_to_send[p].data(),
                nnz_to_send[p].size(),
                MPI_INT,
                p,
                ilu->N + 3,
                MPI_COMM_WORLD,
                &req
            );
            requests.push_back(req);
        }
    }

    for (int p = 0; p < ilu->world_size; ++p) {
        if (!rows_to_recv[p].empty()) {
            nnz_to_recv[p].resize(rows_to_recv[p].size());
            MPI_Request req;
            MPI_Irecv(
                nnz_to_recv[p].data(),
                nnz_to_recv[p].size(),
                MPI_INT,
                p,
                ilu->N + 3,
                MPI_COMM_WORLD,
                &req
            );
            requests.push_back(req);
        }
    }

    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    requests.clear();

    for (int p = 0; p < ilu->world_size; ++p) {
        for (size_t i = 0; i < rows_to_recv[p].size(); ++i) {
            int global_row = rows_to_recv[p][i];
            int nnz = nnz_to_recv[p][i];

            ilu->recv_row_buffers.emplace_back(nnz);

            MPI_Request req;
            MPI_Irecv(
                ilu->recv_row_buffers.back().data(),
                nnz * sizeof(RowElem),
                MPI_BYTE,
                p,
                global_row,
                MPI_COMM_WORLD,
                &req
            );

            ilu->active_recv_requests.push_back(req);
            ilu->request_to_row_idx.push_back(global_row);
            ilu->row_idx_to_request_idx[global_row] =
                ilu->active_recv_requests.size() - 1;
        }
    }
    ilu->count_active_requests = ilu->active_recv_requests.size();
}

bool ILU_row_with_externals(struct ILUFact *ilu, int row_local, int first_idx) {
    auto &LU = ilu->LU;
    auto &idx = ilu->first_col_in_separator_idx[row_local - ilu->num_interior];

    for (; idx < ilu->LU.row_ptr[row_local + 1]; ++idx) {
        int col = LU.col_idx[idx];
        double val = LU.val[idx];

        if (col >= row_local + ilu->global_offset) {
            break; // return true;
        }

        if (utils::is_zero(val)) {
            continue;
        }


        int a_kk_idx = -1;
        std::vector<int> other_cols;
        std::vector<double> other_vals;

        if (col < ilu->global_offset) {
            if (ilu->active_recv_requests[ilu->row_idx_to_request_idx[col]] ==
                MPI_REQUEST_NULL) {
                auto &recived_row =
                    ilu->recv_row_buffers[ilu->row_idx_to_request_idx[col]];
                std::tie(other_cols, other_vals) = RowElem::unpack(recived_row);
            }
            else {
                return false;
            }
        }
        else if (
            (col >= ilu->global_offset &&
             col < ilu->global_offset + ilu->num_interior) ||  // in interior
            (
                ilu->is_separator_ready
                    [col - ilu->global_offset - ilu->num_interior]
            )  // in separator and ready
        ) {
            int local_col = col - ilu->global_offset;
            other_cols.reserve(LU.nnz_in_local_row(local_col));
            other_vals.reserve(LU.nnz_in_local_row(local_col));

            for (int i = LU.row_ptr[local_col]; i < LU.row_ptr[local_col + 1];
                 ++i) {
                other_cols.push_back(LU.col_idx[i]);
                other_vals.push_back(LU.val[i]);
            }
        }
        else {
            return false;
        }

        // Find the diagonal element and factorize
        auto it = std::lower_bound(other_cols.begin(), other_cols.end(), col);
        if (it != other_cols.end() && *it == col) {
            a_kk_idx = std::distance(other_cols.begin(), it);
        }
        else {
            throw std::runtime_error(
                "Expected to find diagonal element in received row"
            );
        }

        LU.val[idx] /= other_vals[a_kk_idx];
        LU.add_mult_row_to_row(
            -LU.val[idx],
            row_local,
            other_vals.data() + a_kk_idx + 1,
            other_cols.data() + a_kk_idx + 1,
            other_vals.size() - a_kk_idx - 1

        );
    }
    return true;
}

std::vector<int> incorporate_received_row(
    struct ILUFact *ilu, int request_idx
) {
    int global_row = ilu->request_to_row_idx[request_idx];

    std::vector<int> ready_rows_loc;

    for (int local_row_sep = ilu->num_interior;
         local_row_sep < ilu->num_rows_local;
         ++local_row_sep) {
        auto first_col_in_separator_idx =
            ilu->first_col_in_separator_idx[local_row_sep - ilu->num_interior];

        if (ilu->LU.col_idx[first_col_in_separator_idx] == global_row) {
            if (ILU_row_with_externals(
                    ilu, local_row_sep, first_col_in_separator_idx
                )) {
                ready_rows_loc.push_back(local_row_sep);
                ilu->is_separator_ready[local_row_sep - ilu->num_interior] = true;
            }
        }
    }

    return ready_rows_loc;
}

void broadcast_new_rows(
    struct ILUFact *ilu, const std::vector<int> &new_ready_rows_loc
) {
    for (auto &local_row : new_ready_rows_loc) {
        int global_row = local_row + ilu->global_offset;
        int nnz = ilu->LU.nnz_in_local_row(local_row);
        std::vector<RowElem> data_to_send(nnz);

        for (int i = ilu->LU.row_ptr[local_row];
             i < ilu->LU.row_ptr[local_row + 1];
             ++i) {
            data_to_send[i - ilu->LU.row_ptr[local_row]] = {
                ilu->LU.col_idx[i], ilu->LU.val[i]
            };
        }

        for (auto &rank : ilu->send_to_ranks[global_row]) {
            MPI_Send(
                data_to_send.data(),
                nnz * sizeof(RowElem),
                MPI_BYTE,
                rank,
                global_row,
                MPI_COMM_WORLD
            );
        }
    }
}

}  // namespace

// ================================================
// ILU library functions
// ================================================

struct ILUFact *ILU_factorize(int N, int nnz, int *row, int *col, double *val) {
    // 3. Podział wierszy na "interior" i "separator"
    // 4. Renumeracja
    // 5. Lokalna faktoryzacja
    struct ILUFact *ilu = new ILUFact();

    MPI_Comm_rank(MPI_COMM_WORLD, &ilu->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ilu->world_size);

    distribute_data(N, nnz, row, col, val, ilu);
    utils::print_local_dense(ilu);
    interior_separator_partition(ilu);
    utils::print_local_dense(ilu);
    share_dependencies(ilu);  // TODO uwspółbierznić
    ILU(ilu->LU, ilu->global_offset, ilu->num_interior);

    std::vector<int> interior_nodes(ilu->num_interior);
    std::iota(interior_nodes.begin(), interior_nodes.end(), 0);
    broadcast_new_rows(ilu, interior_nodes);

    while (ilu->count_active_requests > 0) {
        int indx;
        MPI_Waitany(
            ilu->active_recv_requests.size(),
            ilu->active_recv_requests.data(),
            &indx,
            MPI_STATUS_IGNORE
        );
        ilu->count_active_requests--;

        auto new_ready_rows_loc = incorporate_received_row(ilu, indx);
        if (!new_ready_rows_loc.empty()) {
            broadcast_new_rows(ilu, new_ready_rows_loc);
        }
    }

    utils::print_local_dense(ilu);

    return ilu;
}

auto dist_async_solve_L(struct ILUFact *ilu, const std::vector<double> &b) {
    std::vector<double> y(ilu->num_rows_local, 0);

    // int residue;
    // do {
    //     // Recieve rows
    //     // Send rows
    //
    // while ()

    return y;
}

void ILU_solve(struct ILUFact *ilu, double *b, double *res) {
    // 1. Zastosowanie permutacji do wektora b
    std::vector<double> b_vec(b, b + ilu->num_rows_local);
    // b_vec = utils::permutation::apply_permutation(b_vec, ilu->perm);
    // b_vec = dist_async_solve_L(ilu, b_vec);
    // b_vec = dist_async_solve_U(ilu, b_vec);

    memcpy(res, b_vec.data(), ilu->num_rows_local * sizeof(double));
}

void ILU_multiply(struct ILUFact *ilu, double *b, double *res) {
    std::vector<double> b_vec(b, b + ilu->num_rows_local);
    // b_vec = multiply_U(ilu, b_vec);
    // b_vec = multiply_L(ilu, b_vec);
    memcpy(res, b_vec.data(), ilu->num_rows_local * sizeof(double));
}

void ILU_free(struct ILUFact *ilu) {
    if (ilu != nullptr) {
        delete ilu;
    }
}
