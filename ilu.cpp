#include "ilu.h"

#include <cstdlib>
#include <map>
#include <vector>

struct CSRMatrix {
    int num_rows;
    int num_cols;
    int nnz;
    std::vector<int> row_ptr;   // Size: num_rows + 1
    std::vector<int> col_idx;   // Size: nnz
    std::vector<double> val;    // Size: nnz
    std::vector<int> diag_idx;  // num_rows

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

    static CSRMatrix Identity(int N) {
        CSRMatrix I;
        I.num_rows = N;
        I.nnz = N;
        I.row_ptr.resize(N + 1);
        I.col_idx.resize(N);
        I.val.resize(N, 1.0);

        for (int i = 0; i < N; ++i) {
            I.row_ptr[i] = i;
            I.col_idx[i] = i;
        }
        I.row_ptr[N] = N;

        return I;
    }
};

struct ILUFact {
    int N;
    int rank;
    int world_size;

    int local_N;
    int global_offset;

    int num_interior;
    int num_separator;

    int glob_to_loc_row(int global_row) const {
        return global_row - global_offset;
    }

    int loc_to_glob_row(int local_row) const {
        return local_row + global_offset;
    }

    std::vector<int> perm;
    std::vector<int> inv_perm;

    CSRMatrix LU;  // Unified matrix for both L and U

    std::vector<int> send_to_ranks;
    std::vector<int> recv_from_ranks;
};

static void ILU(CSRMatrix &LU, const int num_rows_from_U_to_factorize) {
    for (int i = 1; i < num_rows_from_U_to_factorize; ++i) {
        for (int k = LU.row_ptr[i]; k < LU.row_ptr[i + 1]; ++k) {
            int row_to_subtract = LU.col_idx[k];

            if (row_to_subtract >= i) {
                break;
            }

            int diag_kk_idx = LU.diag_idx[row_to_subtract];
            LU.val[k] = LU.val[k] / LU.val[diag_kk_idx];

            LU.add_mult_row_to_row(
                -LU.val[k],
                i,
                LU.val.data() + diag_kk_idx + 1,
                LU.col_idx.data() + diag_kk_idx + 1,
                LU.row_ptr[row_to_subtract + 1] - LU.row_ptr[row_to_subtract] -
                    1
            );
        }
    }
}

/**
 * Performs the distributed ILU factorization.
 */
struct ILUFact *ILU_factorize(int N, int nnz, int *row, int *col, double *val) {
    // 1. Inicjalizacja MPI (jeśli jeszcze nie zrobiona)
    // 2. Dystrybucja danych między procesy (tylko rank 0 wysyła)
    // 3. Podział wierszy na "interior" i "separator"
    // 4. Renumeracja
    // 5. Lokalna faktoryzacja

    struct ILUFact *ilu = new ILUFact();
    // Inicjalizacja pól struktury...

    return ilu;
}

/**
 * Solves the linear equation LUx = b.
 */
void ILU_solve(struct ILUFact *ilu, double *b, double *res) {
    // 1. Zastosowanie permutacji do wektora b
    // 2. Rozwiązanie układu z macierzą dolnotrójkątną L (w przód)
    // 3. Rozwiązanie układu z macierzą górnotrójkątną U (wstecz)
    // 4. Komunikacja międzyprocesowa w trakcie rozwiązywania
}

/**
 * Multiplies the vector b by LU.
 */
void ILU_multiply(struct ILUFact *ilu, double *b, double *res) {
    // 1. Mnożenie wektora przez macierz U
    // 2. Mnożenie wyniku przez macierz L
    // 3. Obsługa komunikacji między procesami
}

/**
 * Frees the memory allocated by ILU_factorize.
 */
void ILU_free(struct ILUFact *ilu) {
    if (ilu != nullptr) {
        // Zwolnienie dynamicznie alokowanych pól wewnątrz struktury
        // delete[] ilu->local_L; ...

        delete ilu;
    }
}
