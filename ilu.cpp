// Optymalizacje do napisania:
// - wysyłanie tylko wyrazy po prawej stronie przekątnej
//

#include "ilu.h"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <unistd.h>

const double EPS = 1e-10;
const double FACTORIZE_EPS = 1e-8;
const int FACTORIZE_MAX_SWEEPS = 50;

#define FOR_CSR(matrix_ptr, row_var, idx_var) \
    for (int row_var = 0; row_var < (matrix_ptr)->num_rows; ++row_var) \
        for (int idx_var = (matrix_ptr)->row_ptr[row_var]; idx_var < (matrix_ptr)->row_ptr[row_var + 1]; ++idx_var)

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

struct CommunicationTopology {
    // local_row -> ranks I need to send this row to
    std::vector<std::vector<int>> lcrow_to_ranks_to_send;
    // global_row -> rank from which I need to receive the row
    std::unordered_map<int, int> glbrow_to_rank_to_recv;
    std::unordered_map<int, int> glbrow_row_nnz_to_recv;
    // factorization: interior rows received once; separator rows each sweep
    std::unordered_map<int, int> glbrow_interior_recv;
    std::unordered_map<int, int> glbrow_interior_nnz;
    std::unordered_map<int, int> glbrow_separator_recv;
    std::unordered_map<int, int> glbrow_separator_nnz;
    // packed solve exchange: one MPI message per neighbor rank
    std::unordered_map<int, std::vector<int>> rank_to_glbrows_recv;
    std::unordered_map<int, std::vector<int>> rank_to_localrows_send;
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

namespace {
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
}
} // namespace

struct ILUFact {
    int N;
    int rank;
    int world_size;

    int global_offset;
    int num_rows_local;

    int num_interior;
    int num_separator;

    std::vector<int> local_perm;
    std::vector<int> local_inv_perm;
    std::vector<int> global_perm;

    CSRMatrix LU;  // Unified matrix for both L and U
    CSRMatrix separator_backup;
    std::unordered_map<int, std::vector<RowElem>> external_row_cache;
    std::vector<double> separator_vals_prev;
    int factorization_sweep_count = 0;

    CommunicationTopology lower_rank_topo;
    CommunicationTopology higher_rank_topo;
    std::vector<int> first_row_in_rank;
    std::vector<int> num_rows_in_rank;
    std::vector<int> num_interior_in_rank;

    int glob_to_loc_row(int global_row) const {
        return global_row - global_offset;
    }

    int loc_to_glob_row(int local_row) const {
        return local_row + global_offset;
    }

    void setup(int N, int world_size, int rank) {
        this->N = N;
        this->world_size = world_size;
        this->rank = rank;

        first_row_in_rank.resize(world_size);
        num_rows_in_rank.resize(world_size);

        first_row_in_rank[0] = 0;
        for (int i = 1; i < world_size; ++i) {
            first_row_in_rank[i] = utils::get_first_row_of_process(i, world_size, N);
            num_rows_in_rank[i - 1] = first_row_in_rank[i] - first_row_in_rank[i - 1];
        }
        num_rows_in_rank[world_size - 1] = N - first_row_in_rank[world_size - 1];

        this->global_offset = first_row_in_rank[rank];
        this->num_rows_local = num_rows_in_rank[rank];
    }

};

namespace {

// ========= UTILITIES =========

namespace utils {

void print_local_dense(const struct ILUFact *ilu) {
    const int precision = 2;
    const int width = 5;
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
                        std::cout << std::setw(width) << "*";
                        current_col++;
                    }
                    std::cout << std::setw(width) << std::fixed
                              << std::setprecision(precision) << mat.val[idx];
                    current_col++;
                }
                while (current_col < ilu->N) {
                    std::cout << std::setw(width) << "*";
                    current_col++;
                }
                std::cout << "\n";
            }
            std::cout << std::flush;

            usleep(10000);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        usleep(10000);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void sort_row(CSRMatrix &LU, int row) {
    int row_start = LU.row_ptr[row];
    int row_end = LU.row_ptr[row + 1];
    std::vector<std::pair<int, double>> row_elems;
    row_elems.reserve(row_end - row_start);
    for (int i = row_start; i < row_end; ++i) {
        row_elems.push_back({LU.col_idx[i], LU.val[i]});
    }
    std::sort(row_elems.begin(), row_elems.end(), [&](const std::pair<int, double> &a, const std::pair<int, double> &b) {
        return a.first < b.first;
    }); 
    for (int i = row_start; i < row_end; ++i) {
        LU.col_idx[i] = row_elems[i - row_start].first;
        LU.val[i] = row_elems[i - row_start].second;
    }
}

// ========= PERMUTATION UTILITIES =========

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

void permute_rows(CSRMatrix &LU, const std::vector<int> &inv_perm) {
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

        for (int idx = row_start; idx < row_end; ++idx) {
            perm_col_idx.push_back(LU.col_idx[idx]);
            perm_val.push_back(LU.val[idx]);
        }

        perm_row_ptr.push_back(static_cast<int>(perm_col_idx.size()));
    }

    LU.row_ptr = std::move(perm_row_ptr);
    LU.col_idx = std::move(perm_col_idx);
    LU.val = std::move(perm_val);
    LU.nnz = static_cast<int>(LU.col_idx.size());
}

void permute_columns(CSRMatrix &LU, const std::vector<int> &global_perm) {
    for (int i = 0; i < LU.num_rows; ++i) {
        for (int j = LU.row_ptr[i]; j < LU.row_ptr[i + 1]; ++j) {
            LU.col_idx[j] = global_perm[LU.col_idx[j]];
        }
        sort_row(LU, i);
    }
}

}  // namespace permutation
}  // namespace utils

//
// ========= Distribute data functions =========
//

void distribute_data(
    int N, int nnz, const int *row, const int *col, const double *val, struct ILUFact *ilu
) {
    CSRMatrix &received_matrix = ilu->LU;

    auto get_first_row_of_process = [&](int rank) {
        return utils::get_first_row_of_process(rank, ilu->world_size, N);
    };

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    ilu->setup(N, ilu->world_size, ilu->rank);

    received_matrix.num_rows = ilu->num_rows_local;
    received_matrix.num_cols = N;
    received_matrix.row_ptr.resize(received_matrix.num_rows + 1);

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

        received_matrix.nnz = col_idx_sendcounts[0];
        std::copy(
            A.row_ptr.begin() + first_row,
            A.row_ptr.begin() + last_row + 2,
            received_matrix.row_ptr.begin()
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
            &received_matrix.nnz,
            1,
            MPI_INT,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
        MPI_Recv(
            received_matrix.row_ptr.data(),
            received_matrix.num_rows + 1,
            MPI_INT,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
    }

    received_matrix.col_idx.resize(received_matrix.nnz);
    received_matrix.val.resize(received_matrix.nnz);

    MPI_Scatterv(
        ilu->rank == 0 ? A.col_idx.data() : nullptr,
        col_idx_sendcounts.data(),
        col_idx_displacements.data(),
        MPI_INT,
        received_matrix.col_idx.data(),
        received_matrix.nnz,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    MPI_Scatterv(
        ilu->rank == 0 ? A.val.data() : nullptr,
        col_idx_sendcounts.data(),
        col_idx_displacements.data(),
        MPI_DOUBLE,
        received_matrix.val.data(),
        received_matrix.nnz,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    int offset = received_matrix.row_ptr[0];
    for (auto &rp : received_matrix.row_ptr) {
        rp -= offset;
    }
}

//
// ========= Partition and permutation functions =========
// 

void interior_separator_partition(struct ILUFact *ilu) {
    ilu->local_inv_perm.resize(ilu->num_rows_local);

    std::vector<int> interior_rows;
    std::vector<int> separator_rows;

    CSRMatrix &LU = ilu->LU;

    for (int row_local = 0; row_local < LU.num_rows; ++row_local) {
        bool is_separator = false;
        for (int idx = LU.row_ptr[row_local]; idx < LU.row_ptr[row_local + 1];
             ++idx) {
            if (LU.col_idx[idx] < ilu->global_offset) {
                is_separator = true;
                break;
            }
        }

        if (is_separator) {
            separator_rows.push_back(row_local);
        }
        else {
            interior_rows.push_back(row_local);
        }
    }

    ilu->num_interior = interior_rows.size();
    ilu->num_separator = separator_rows.size();
    interior_rows.insert(
        interior_rows.end(), separator_rows.begin(), separator_rows.end()
    );

    ilu->local_inv_perm = std::move(interior_rows);
    ilu->local_perm = utils::permutation::inverse_permutation(ilu->local_inv_perm);

    utils::permutation::permute_rows(LU, ilu->local_inv_perm);
}

void share_permutation(struct ILUFact *ilu) {
    ilu->global_perm.resize(ilu->N);
    MPI_Allgatherv(
        ilu->local_perm.data(),
        ilu->num_rows_local,
        MPI_INT,
        ilu->global_perm.data(),
        ilu->num_rows_in_rank.data(),
        ilu->first_row_in_rank.data(),
        MPI_INT,
        MPI_COMM_WORLD
    );
    for (int r = 0; r < ilu->world_size; ++r) {
        for (int i = ilu->first_row_in_rank[r]; i < ilu->first_row_in_rank[r] + ilu->num_rows_in_rank[r]; ++i) {
            ilu->global_perm[i] += ilu->first_row_in_rank[r];
        }
    }
}

//
// ========= Communication functions =========
//

const MPI_Datatype &row_elem_mpi_type() {
    static MPI_Datatype type = MPI_DATATYPE_NULL;
    if (type == MPI_DATATYPE_NULL) {
        int count = 2;
        int blocklengths[2] = {1, 1};
        MPI_Datatype types[2] = {MPI_INT, MPI_DOUBLE};
        MPI_Aint displacements[2];
        displacements[0] = offsetof(RowElem, col);
        displacements[1] = offsetof(RowElem, val);
        MPI_Type_create_struct(
            count, blocklengths, displacements, types, &type
        );
        MPI_Type_commit(&type);
    }
    return type;
}

auto calculate_needed_rows_from_other_ranks(struct ILUFact *ilu) {
    std::vector<std::set<int>> needed_rows_from_rank(ilu->world_size);

    for (int row_local = 0; row_local < ilu->num_rows_local;
            ++row_local) {
        for (int idx = ilu->LU.row_ptr[row_local];
                idx < ilu->LU.row_ptr[row_local + 1];
                ++idx) {
            int global_col = ilu->LU.col_idx[idx];

            int owner_rank = utils::get_owner_rank(global_col, ilu->world_size, ilu->N);
            if (owner_rank != ilu->rank) {
                needed_rows_from_rank[owner_rank].insert(global_col);
            }
        }
    }
    return needed_rows_from_rank;
}

auto share_needed_rows_nnz(struct ILUFact *ilu, const std::vector<std::set<int>> &needed_rows_from_rank) {
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

    return std::make_pair(requests_count_to_send, requests_count_to_recv);
}

auto share_needed_rows(
    struct ILUFact *ilu,
    const std::vector<std::set<int>> &needed_rows_from_rank, 
    const std::vector<int> &requests_count_to_send,
    const std::vector<int> &requests_count_to_recv
) {
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
    return std::make_pair(rows_to_recv, rows_to_send);
}

auto share_nnz(
    struct ILUFact *ilu,
    const std::vector<std::vector<int>> &rows_to_recv,
    const std::vector<std::vector<int>> &rows_to_send
) {
    std::vector<std::vector<int>> nnz_to_send(ilu->world_size);
    std::vector<std::vector<int>> nnz_to_recv(ilu->world_size);
    std::vector<MPI_Request> requests;

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

    return std::make_pair(nnz_to_recv, nnz_to_send);
}

void build_solve_exchange_groups(CommunicationTopology &topo) {
    topo.rank_to_glbrows_recv.clear();
    topo.rank_to_localrows_send.clear();
    for (const auto &[global_row, src_rank] : topo.glbrow_to_rank_to_recv) {
        topo.rank_to_glbrows_recv[src_rank].push_back(global_row);
    }
    for (auto &[rank, rows] : topo.rank_to_glbrows_recv) {
        std::sort(rows.begin(), rows.end());
    }
    for (int local_row = 0; local_row < (int)topo.lcrow_to_ranks_to_send.size();
         ++local_row) {
        for (int dest_rank : topo.lcrow_to_ranks_to_send[local_row]) {
            topo.rank_to_localrows_send[dest_rank].push_back(local_row);
        }
    }
    for (auto &[rank, rows] : topo.rank_to_localrows_send) {
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    }
}

int solve_vector_tag(int N, int sender_rank) { return N + 1000 + sender_rank; }

void split_recv_maps_by_row_type(ILUFact *ilu, CommunicationTopology &topo) {
    topo.glbrow_interior_recv.clear();
    topo.glbrow_interior_nnz.clear();
    topo.glbrow_separator_recv.clear();
    topo.glbrow_separator_nnz.clear();

    for (const auto &[global_row, src_rank] : topo.glbrow_to_rank_to_recv) {
        int local_on_src = global_row - ilu->first_row_in_rank[src_rank];
        int nnz = topo.glbrow_row_nnz_to_recv.at(global_row);
        if (local_on_src < ilu->num_interior_in_rank[src_rank]) {
            topo.glbrow_interior_recv[global_row] = src_rank;
            topo.glbrow_interior_nnz[global_row] = nnz;
        } else {
            topo.glbrow_separator_recv[global_row] = src_rank;
            topo.glbrow_separator_nnz[global_row] = nnz;
        }
    }
}

void setup_communication_topology(struct ILUFact *ilu) {
    ilu->num_interior_in_rank.resize(ilu->world_size);
    MPI_Allgather(
        &ilu->num_interior,
        1,
        MPI_INT,
        ilu->num_interior_in_rank.data(),
        1,
        MPI_INT,
        MPI_COMM_WORLD
    );

    // Calculate my dependencies
    auto needed_rows_from_rank = calculate_needed_rows_from_other_ranks(ilu);
    auto [requests_count_to_send, requests_count_to_recv] = share_needed_rows_nnz(ilu, needed_rows_from_rank);

    auto [rows_to_recv, rows_to_send] = share_needed_rows(ilu, needed_rows_from_rank, requests_count_to_send, requests_count_to_recv);
    auto [nnz_to_recv, nnz_to_send] = share_nnz(ilu, rows_to_recv, rows_to_send);

    ilu->lower_rank_topo.lcrow_to_ranks_to_send.resize(ilu->num_rows_local);
    ilu->higher_rank_topo.lcrow_to_ranks_to_send.resize(ilu->num_rows_local);
    for (int p = 0; p < ilu->world_size; ++p) {
        for (int global_row : rows_to_send[p]) {
            int local_row = global_row - ilu->global_offset;
            if (p > ilu->rank) {
                ilu->lower_rank_topo.lcrow_to_ranks_to_send[local_row].push_back(p);
            }
            else if (p < ilu->rank) {
                ilu->higher_rank_topo.lcrow_to_ranks_to_send[local_row].push_back(p);
            }
        }
        for (size_t i = 0; i < rows_to_recv[p].size(); ++i) {
            int global_row = rows_to_recv[p][i];
            int nnz = nnz_to_recv[p][i];
            if (p < ilu->rank) {
                ilu->lower_rank_topo.glbrow_to_rank_to_recv[global_row] = p;
                ilu->lower_rank_topo.glbrow_row_nnz_to_recv[global_row] = nnz;
            }
            else if (p > ilu->rank) {
                ilu->higher_rank_topo.glbrow_to_rank_to_recv[global_row] = p;
                ilu->higher_rank_topo.glbrow_row_nnz_to_recv[global_row] = nnz;
            }
        }
    }
    split_recv_maps_by_row_type(ilu, ilu->lower_rank_topo);
    split_recv_maps_by_row_type(ilu, ilu->higher_rank_topo);
    build_solve_exchange_groups(ilu->lower_rank_topo);
    build_solve_exchange_groups(ilu->higher_rank_topo);
}


void share_dependencies(struct ILUFact *ilu) {
    setup_communication_topology(ilu);

    auto &topo = ilu->lower_rank_topo;
    for (const auto &[global_row, src_rank] : topo.glbrow_to_rank_to_recv) {
        ilu->external_row_cache[global_row].resize(
            topo.glbrow_row_nnz_to_recv.at(global_row)
        );
    }
}

void backup_separator_rows(ILUFact *ilu) {
    auto &LU = ilu->LU;
    auto &backup = ilu->separator_backup;

    backup.num_rows = ilu->num_separator;
    backup.num_cols = LU.num_cols;
    backup.nnz = 0;
    for (int row = ilu->num_interior; row < LU.num_rows; ++row) {
        backup.nnz += LU.nnz_in_local_row(row);
    }

    backup.row_ptr.assign(backup.num_rows + 1, 0);
    backup.col_idx.resize(backup.nnz);
    backup.val.resize(backup.nnz);

    int dst = 0;
    for (int i = 0; i < ilu->num_separator; ++i) {
        int src_row = ilu->num_interior + i;
        for (int j = LU.row_ptr[src_row]; j < LU.row_ptr[src_row + 1]; ++j) {
            backup.col_idx[dst] = LU.col_idx[j];
            backup.val[dst] = LU.val[j];
            ++dst;
        }
        backup.row_ptr[i + 1] = dst;
    }
}

void restore_separator_rows(ILUFact *ilu) {
    auto &LU = ilu->LU;
    auto &backup = ilu->separator_backup;

    for (int i = 0; i < ilu->num_separator; ++i) {
        int dst_row = ilu->num_interior + i;
        int src_start = backup.row_ptr[i];
        int dst_start = LU.row_ptr[dst_row];
        int nnz = backup.row_ptr[i + 1] - backup.row_ptr[i];
        for (int k = 0; k < nnz; ++k) {
            LU.val[dst_start + k] = backup.val[src_start + k];
        }
    }
}

void send_factorized_rows(ILUFact *ilu, const std::vector<int> &local_rows) {
    static const MPI_Datatype &row_mpi_type = row_elem_mpi_type();
    for (int local_row : local_rows) {
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

        for (int rank : ilu->lower_rank_topo.lcrow_to_ranks_to_send[local_row]) {
            MPI_Send(
                data_to_send.data(),
                nnz,
                row_mpi_type,
                rank,
                global_row,
                MPI_COMM_WORLD
            );
        }
    }
}

void receive_rows_from_map(
    ILUFact *ilu,
    const std::unordered_map<int, int> &glbrow_to_rank,
    const std::unordered_map<int, int> &glbrow_nnz
) {
    static const MPI_Datatype &row_mpi_type = row_elem_mpi_type();

    for (const auto &[global_row, src_rank] : glbrow_to_rank) {
        MPI_Recv(
            ilu->external_row_cache[global_row].data(),
            glbrow_nnz.at(global_row),
            row_mpi_type,
            src_rank,
            global_row,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
    }
}

void receive_interior_rows_once(ILUFact *ilu) {
    auto &topo = ilu->lower_rank_topo;
    receive_rows_from_map(
        ilu, topo.glbrow_interior_recv, topo.glbrow_interior_nnz
    );
}

void receive_separator_rows(ILUFact *ilu) {
    auto &topo = ilu->lower_rank_topo;
    receive_rows_from_map(
        ilu, topo.glbrow_separator_recv, topo.glbrow_separator_nnz
    );
}

void send_rows_in_range(ILUFact *ilu, int row_begin, int row_end) {
    std::vector<int> rows_to_send;
    rows_to_send.reserve(ilu->num_rows_local);
    for (int local_row = row_begin; local_row < row_end; ++local_row) {
        if (!ilu->lower_rank_topo.lcrow_to_ranks_to_send[local_row].empty()) {
            rows_to_send.push_back(local_row);
        }
    }
    if (!rows_to_send.empty()) {
        send_factorized_rows(ilu, rows_to_send);
    }
}

void send_interior_rows(ILUFact *ilu) {
    send_rows_in_range(ilu, 0, ilu->num_interior);
}

void send_separator_rows(ILUFact *ilu) {
    send_rows_in_range(ilu, ilu->num_interior, ilu->num_rows_local);
}

void exchange_interior_rows(ILUFact *ilu) {
    send_interior_rows(ilu);
    receive_interior_rows_once(ilu);
}

//
// ========= Factorization functions =========
// 

void ILU(
    CSRMatrix &LU, int global_offset, const int num_rows_from_U_to_factorize
) {
    for (int i = 1; i < num_rows_from_U_to_factorize; ++i) {
        for (int k = LU.row_ptr[i]; k < LU.row_ptr[i + 1]; ++k) {
            int row_to_subtract_loc = LU.col_idx[k] - global_offset;

            if (LU.col_idx[k] >= global_offset + i) {
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

void factorize_interior_block(ILUFact *ilu) {
    ILU(ilu->LU, ilu->global_offset, ilu->num_interior);
}

bool get_pivot_row_data(
    ILUFact *ilu,
    int global_col,
    std::vector<int> &cols,
    std::vector<double> &vals
) {
    auto &LU = ilu->LU;

    if (global_col < ilu->global_offset) {
        auto it = ilu->external_row_cache.find(global_col);
        if (it == ilu->external_row_cache.end()) {
            return false;
        }
        std::tie(cols, vals) = RowElem::unpack(it->second);
        return true;
    }

    int local_col = global_col - ilu->global_offset;
    cols.clear();
    vals.clear();
    cols.reserve(LU.nnz_in_local_row(local_col));
    vals.reserve(LU.nnz_in_local_row(local_col));
    for (int i = LU.row_ptr[local_col]; i < LU.row_ptr[local_col + 1]; ++i) {
        cols.push_back(LU.col_idx[i]);
        vals.push_back(LU.val[i]);
    }
    return true;
}

void factorize_separator_row(ILUFact *ilu, int row_local) {
    auto &LU = ilu->LU;

    for (int idx = LU.row_ptr[row_local]; idx < LU.row_ptr[row_local + 1]; ++idx) {
        int col = LU.col_idx[idx];
        if (col >= row_local + ilu->global_offset) {
            break;
        }
        if (utils::is_zero(LU.val[idx])) {
            continue;
        }

        std::vector<int> other_cols;
        std::vector<double> other_vals;
        if (!get_pivot_row_data(ilu, col, other_cols, other_vals)) {
            throw std::runtime_error("Pivot row unavailable during separator factorization");
        }

        auto it = std::lower_bound(other_cols.begin(), other_cols.end(), col);
        if (it == other_cols.end() || *it != col) {
            throw std::runtime_error(
                "Expected to find diagonal element in received row"
            );
        }
        int a_kk_idx = std::distance(other_cols.begin(), it);

        LU.val[idx] /= other_vals[a_kk_idx];
        LU.add_mult_row_to_row(
            -LU.val[idx],
            row_local,
            other_vals.data() + a_kk_idx + 1,
            other_cols.data() + a_kk_idx + 1,
            other_vals.size() - a_kk_idx - 1
        );
    }
}

void factorize_separator_block(ILUFact *ilu) {
    for (int sep = ilu->num_interior; sep < ilu->num_rows_local; ++sep) {
        factorize_separator_row(ilu, sep);
    }
}

void snapshot_separator_vals(ILUFact *ilu, std::vector<double> &out) {
    out.clear();
    auto &LU = ilu->LU;
    for (int row = ilu->num_interior; row < LU.num_rows; ++row) {
        for (int idx = LU.row_ptr[row]; idx < LU.row_ptr[row + 1]; ++idx) {
            out.push_back(LU.val[idx]);
        }
    }
}

double max_separator_change(ILUFact *ilu, const std::vector<double> &prev) {
    double max_diff = 0.0;
    size_t k = 0;
    auto &LU = ilu->LU;
    for (int row = ilu->num_interior; row < LU.num_rows; ++row) {
        for (int idx = LU.row_ptr[row]; idx < LU.row_ptr[row + 1]; ++idx) {
            if (k < prev.size()) {
                max_diff = std::max(max_diff, std::abs(LU.val[idx] - prev[k]));
            }
            ++k;
        }
    }
    return max_diff;
}

bool check_factorization_converged(
    ILUFact *ilu,
    const std::vector<double> &prev,
    double tol,
    MPI_Comm sep_comm
) {
    double local_max = max_separator_change(ilu, prev);
    double global_max = 0.0;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, sep_comm);
    return global_max < tol;
}

bool run_separator_sweep(
    ILUFact *ilu, std::vector<double> &prev_vals, int sweep, MPI_Comm sep_comm
) {
    send_separator_rows(ilu);
    receive_separator_rows(ilu);
    restore_separator_rows(ilu);
    factorize_separator_block(ilu);

    bool converged = check_factorization_converged(
        ilu, prev_vals, FACTORIZE_EPS, sep_comm
    );
    snapshot_separator_vals(ilu, prev_vals);
    return converged;
}

void factorize_separators_sweeps(ILUFact *ilu) {
    if (ilu->num_separator == 0) {
        return;
    }

    MPI_Comm sep_comm;
    MPI_Comm_split(MPI_COMM_WORLD, 0, ilu->rank, &sep_comm);

    snapshot_separator_vals(ilu, ilu->separator_vals_prev);

    bool converged = false;
    int sweep = 0;
    const int max_sweeps = std::min(
        std::max(ilu->world_size, 1), FACTORIZE_MAX_SWEEPS
    );

    while (!converged && sweep < max_sweeps) {
        converged = run_separator_sweep(
            ilu, ilu->separator_vals_prev, sweep, sep_comm
        );
        ++sweep;
    }

    ilu->factorization_sweep_count = sweep;
    MPI_Comm_free(&sep_comm);
}

//
// ========= Solve functions =========
//

enum class SolveType {
    L,
    U
};

auto solve_L(struct CSRMatrix &LU, const std::vector<double> &b, int global_offset) {
    std::vector<double> x(LU.num_rows, 0);
    for (int i = 0; i < LU.num_rows; ++i) {
        double cum = 0;
        for (int j = LU.row_ptr[i]; j < LU.row_ptr[i + 1]; ++j) {
            int col_local = LU.col_idx[j] - global_offset;
            if (col_local < 0) {
                continue;
            }
            if (col_local < i) {
                cum += LU.val[j] * x[col_local];
            } else { 
                break;
            }
        }
        x[i] = b[i] - cum;
    }
    return x;
}

auto solve_U(struct CSRMatrix &LU, const std::vector<double> &b, int global_offset) {
    std::vector<double> x(LU.num_rows, 0);
    for (int i = LU.num_rows - 1; i >= 0; --i) {
        double cum = 0;
        int diag_idx = -1;
        for (int j = LU.row_ptr[i + 1] - 1; j >= LU.row_ptr[i]; --j) {
            int col_local = LU.col_idx[j] - global_offset;
            if (col_local >= LU.num_rows) {
                continue;
            }
            if (col_local > i) {
                cum += LU.val[j] * x[col_local];
            } 
            else if (col_local == i) {
                diag_idx = j;
                break;
            } else {
                throw std::runtime_error("Expected to find diagonal in row while solving U");
            }
        }
        x[i] = (b[i] - cum) / LU.val[diag_idx];
    }
    return x;
}

auto share_vector(struct ILUFact *ilu, const std::vector<double> &vec, const CommunicationTopology &topo) {
    std::unordered_map<int, double> external_vec;
    std::vector<MPI_Request> requests;
    std::unordered_map<int, std::vector<double>> recv_bufs;
    std::unordered_map<int, std::vector<double>> send_bufs;

    for (const auto &[src_rank, glb_rows] : topo.rank_to_glbrows_recv) {
        auto &buf = recv_bufs[src_rank];
        buf.resize(glb_rows.size());
        MPI_Request req;
        MPI_Irecv(
            buf.data(),
            static_cast<int>(buf.size()),
            MPI_DOUBLE,
            src_rank,
            solve_vector_tag(ilu->N, src_rank),
            MPI_COMM_WORLD,
            &req
        );
        requests.push_back(req);
    }

    for (const auto &[dest_rank, local_rows] : topo.rank_to_localrows_send) {
        auto &buf = send_bufs[dest_rank];
        buf.resize(local_rows.size());
        for (size_t i = 0; i < local_rows.size(); ++i) {
            buf[i] = vec[local_rows[i]];
        }
        MPI_Request req;
        MPI_Isend(
            buf.data(),
            static_cast<int>(buf.size()),
            MPI_DOUBLE,
            dest_rank,
            solve_vector_tag(ilu->N, ilu->rank),
            MPI_COMM_WORLD,
            &req
        );
        requests.push_back(req);
    }

    if (!requests.empty()) {
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }

    for (const auto &[src_rank, glb_rows] : topo.rank_to_glbrows_recv) {
        const auto &buf = recv_bufs.at(src_rank);
        for (size_t i = 0; i < glb_rows.size(); ++i) {
            external_vec[glb_rows[i]] = buf[i];
        }
    }
    return external_vec;
}

auto dist_async_solve(struct ILUFact *ilu, const std::vector<double> &b, SolveType solve_type) {
    std::vector<double> y;
    switch (solve_type) {
        case SolveType::L:
            y = solve_L(ilu->LU, b, ilu->global_offset);
            break;
        case SolveType::U:
            y = solve_U(ilu->LU, b, ilu->global_offset);
            break;
    }

    int converged = 1;
    int all_converged = 0;

    do {
        auto external_vec = share_vector(
            ilu,
            y,
            solve_type == SolveType::L ? ilu->lower_rank_topo : ilu->higher_rank_topo
        );

        std::vector<double> Ey_ext(ilu->num_rows_local, 0);
        for (int loc_row = 0; loc_row < ilu->num_rows_local; ++loc_row) {
            for (int idx = ilu->LU.row_ptr[loc_row]; idx < ilu->LU.row_ptr[loc_row + 1]; ++idx) {
                int global_col = ilu->LU.col_idx[idx];
                if (solve_type == SolveType::L) {
                    if (global_col < ilu->global_offset) {
                        Ey_ext[loc_row] += external_vec[global_col] * ilu->LU.val[idx];
                    }
                }
                else {
                    if (global_col >= ilu->global_offset + ilu->LU.num_rows) {
                        Ey_ext[loc_row] += external_vec[global_col] * ilu->LU.val[idx];
                    }
                }
            }
        }

        for (int i = 0; i < (int)Ey_ext.size(); ++i) {
            Ey_ext[i] = b[i] - Ey_ext[i];
        }

        std::vector<double> y_new;
        switch (solve_type) {
            case SolveType::L:
                y_new = solve_L(ilu->LU, Ey_ext, ilu->global_offset);
                break;
            case SolveType::U:
                y_new = solve_U(ilu->LU, Ey_ext, ilu->global_offset);
                break;
        }

        converged = 1;
        for (int i = 0; i < (int)y_new.size(); ++i) {
            if (std::abs(y_new[i] - y[i]) > EPS) {
                converged = 0;
                break;
            }
        }

        y = y_new;
    } while (
        MPI_Allreduce(&converged, &all_converged, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD),
        !all_converged
    );

    return y;
}

}  // namespace

// ================================================
// ILU library functions
// ================================================

struct ILUFact* ILU_factorize(int N, int nnz, const int* row, const int* col, const double* val) {
    struct ILUFact *ilu = new ILUFact();

    MPI_Comm_rank(MPI_COMM_WORLD, &ilu->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ilu->world_size);
    
    distribute_data(N, nnz, row, col, val, ilu);
    //utils::print_local_dense(ilu);
    interior_separator_partition(ilu);
    share_permutation(ilu);
    utils::permutation::permute_columns(ilu->LU, ilu->global_perm);

    share_dependencies(ilu);

    backup_separator_rows(ilu);

    factorize_interior_block(ilu);
    exchange_interior_rows(ilu);

    utils::print_local_dense(ilu);
    factorize_separators_sweeps(ilu);   
    utils::print_local_dense(ilu);     
    MPI_Barrier(MPI_COMM_WORLD);

    return ilu;
}

void ILU_solve(struct ILUFact *ilu, const double *b, double *res) {
    std::vector<double> b_vec(b, b + ilu->num_rows_local);
    b_vec = utils::permutation::apply_permutation(b_vec, ilu->local_inv_perm);
    b_vec = dist_async_solve(ilu, b_vec, SolveType::L);
    b_vec = dist_async_solve(ilu, b_vec, SolveType::U);
    b_vec = utils::permutation::apply_permutation(b_vec, ilu->local_perm);

    memcpy(res, b_vec.data(), ilu->num_rows_local * sizeof(double));
}

void ILU_multiply(struct ILUFact *ilu, const double *b, double *res) {
    std::vector<double> b_vec(b, b + ilu->num_rows_local);
    std::vector<double> result(ilu->num_rows_local, 0);

    // TO DO sprawdzić kolejność permutacji 
    b_vec = utils::permutation::apply_permutation(b_vec, ilu->local_inv_perm);
    
    auto ext_higher = share_vector(ilu, b_vec, ilu->higher_rank_topo);

    // multiply by U
    FOR_CSR(&ilu->LU, local_row, idx) {
        int global_col = ilu->LU.col_idx[idx];
        if (global_col >= ilu->global_offset + ilu->LU.num_rows) {
            result[local_row] += ext_higher[global_col] * ilu->LU.val[idx];
        } else if (global_col >= local_row + ilu->global_offset) {
            result[local_row] += ilu->LU.val[idx] * b_vec[global_col - ilu->global_offset];
        }
    }

    b_vec = result;
    result.assign(ilu->num_rows_local, 0);
    auto ext_lower = share_vector(ilu, b_vec, ilu->lower_rank_topo);

    // multiply by L
    FOR_CSR(&ilu->LU, local_row, idx) {
        int global_col = ilu->LU.col_idx[idx];
        if (global_col < ilu->global_offset) {
            result[local_row] += ext_lower[global_col] * ilu->LU.val[idx];
        } else if (global_col < local_row + ilu->global_offset) {
            result[local_row] += ilu->LU.val[idx] * b_vec[global_col - ilu->global_offset];
        }
        else if (global_col == local_row + ilu->global_offset) {
            result[local_row] += 1 * b_vec[global_col - ilu->global_offset];
        }
    }

    result = utils::permutation::apply_permutation(result, ilu->local_perm);
    memcpy(res, result.data(), ilu->num_rows_local * sizeof(double));
}

void ILU_free(struct ILUFact *ilu) {
    if (ilu != nullptr) {
        delete ilu;
    }
}
