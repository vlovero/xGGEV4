#include "benchmark/benchmark.h"
#include "fmt/core.h"
#include "tgevc3.h"
#include "xggev4.h"
#include <algorithm>
#include <random>
#include <vector>

extern "C" {
    int openblas_get_num_threads();
    void openblas_set_num_threads_(int *);
    void dggev3_(const char *jobvl, const char *jobvr, const int *n, double *a, const int *lda, double *b, const int *ldb, double *alphar, double *alphai, double *beta, double *vl, const int *ldvl, double *vr, const int *ldvr, double *work, const int *lwork, int *info);
    void dggev_(const char *jobvl, const char *jobvr, const int *n, double *a, const int *lda, double *b, const int *ldb, double *alphar, double *alphai, double *beta, double *vl, const int *ldvl, double *vr, const int *ldvr, double *work, const int *lwork, int *info);
    void dgghd3_(const char *compq, const char *compz, const int *n, const int *ilo, const int *ihi, double *A, const int *ldA, double *B, const int *ldB, double *Q, const int *ldQ, double *Z, const int *ldZ, double *work, const int *lwork, int *info);
    void dtgevc_(const char *side, const char *howmny, const int *select, const int *n, const double *s, const int *lds, const double *p, const int *ldp, double *vl, const int *ldvl, double *vr, const int *ldvr, const int *mm, int *m, double *work, int *info);
    void dtrevc_(const char *side, const char *howmny, const int *select, const int *n, const double *t, const int *ldt, double *vl, const int *ldvl, double *vr, const int *ldvr, const int *mm, int *m, double *work, int *info);
    void dtrevc3_(const char *side, const char *howmny, const int *select, const int *n, const double *t, const int *ldt, double *vl, const int *ldvl, double *vr, const int *ldvr, const int *mm, int *m, double *work, const int *lwork, int *info);
}

extern void dorgqr(const int m, const int n, const int k, double *A, const int lda, const double *tau, double *work, const int lwork, int *info);
extern void dormqr(const char *side, const char *trans, const int m, const int n, const int k, const double *A, const int lda, const double *tau, double *C, const int ldc, double *work, const int lwork, int *info);
extern void dgeqrf(const int m, const int n, double *A, const int lda, double *tau, double *work, const int lwork, int *info);
extern void dgghd4(const char *compq, const char *compz, const ptrdiff_t n, const int ilo, const int ihi, double *A, const ptrdiff_t ldA, double *B, const ptrdiff_t ldB, double *Q, const ptrdiff_t ldQ, double *Z, const ptrdiff_t ldZ, double *work, int lwork, int *info);
extern void dggprp(const bool wantq, const bool wantz, const int n, double *A, const int ldA, double *B, const int ldB, double *Q, const int ldQ, double *Z, const int ldZ, double *work, const int lwork, int *jpvt, int *info);


class OpenBLASThreadContext
{
    int m_nwanted;
    int m_ndefault;

public:
    OpenBLASThreadContext() = delete;
    OpenBLASThreadContext(const int nwanted) : m_nwanted(nwanted), m_ndefault(openblas_get_num_threads())
    {
        openblas_set_num_threads_(&m_nwanted);
    }

    ~OpenBLASThreadContext()
    {
        openblas_set_num_threads_(&m_ndefault);
    }
};

static std::vector<double> eye(const ptrdiff_t n)
{
    std::vector<double> I(n * n);
    for (ptrdiff_t i = 0; i < n; i++) {
        for (ptrdiff_t j = 0; j < n; j++) {
            I[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    return I;
}

static void generate_generalized_real_triangular(int n, std::vector<double> &S, std::vector<double> &P, std::vector<double> &alphar, std::vector<double> &alphai, std::vector<double> &beta)
{
    S.assign(n * n, 0.0);
    P.assign(n * n, 0.0);
    alphar.assign(n, 0.0);
    alphai.assign(n, 0.0);
    beta.assign(n, 0.0);

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i <= j; ++i) {
            S[i + j * n] = dist(gen);
            P[i + j * n] = dist(gen);
        }
        double p_real = dist(gen);
        P[j + j * n] = p_real + (p_real >= 0 ? 1.0 : -1.0);

        alphar[j] = S[j + j * n];
        beta[j] = P[j + j * n];
    }
}

static void generate_standard_real_triangular(int n, std::vector<double> &T)
{
    T.assign(n * n, 0.0);
    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i <= j; ++i) {
            T[i + j * n] = dist(gen);
        }
    }
}

static void generate_random_matrix(std::vector<double> &A, const int seed)
{
    std::default_random_engine engine(seed);
    std::uniform_real_distribution<double> dist(0, 1);
    std::generate(A.begin(), A.end(), [&]() { return dist(engine); });
}

static void generate_random_matrix(double *x, const ptrdiff_t nx, const int seed)
{
    std::default_random_engine engine(seed);
    std::uniform_real_distribution<double> dist;

    std::generate(x, x + nx, [&]() { return dist(engine); });
}

static void BM_dggev3(benchmark::State &state)
{
    int n = state.range(0);
    int r = state.range(1);
    double work_opt;
    int lwork = -1;
    int info;
    OpenBLASThreadContext __openblas_context(r);
    std::vector<double> alphar(n);
    std::vector<double> alphai(n);
    std::vector<double> beta(n);
    std::vector<double> A_start(n * n);
    std::vector<double> B_start(n * n);
    std::vector<double> A(n * n);
    std::vector<double> B(n * n);
    generate_random_matrix(A_start, 0);
    generate_random_matrix(B_start, 1);
    int ldA = n;
    int ldB = n;
    dggev3_("N", "N", &n, A.data(), &ldA, B.data(), &ldB, alphar.data(), alphai.data(), beta.data(), NULL, &n, NULL, &n, &work_opt, &lwork, &info);

    double *work = new double[(int)work_opt];
    lwork = (int)work_opt;
    for (auto _ : state) {
        memcpy(A.data(), A_start.data(), n * n * sizeof(double));
        memcpy(B.data(), B_start.data(), n * n * sizeof(double));
        dggev3_("N", "N", &n, A.data(), &ldA, B.data(), &ldB, alphar.data(), alphai.data(), beta.data(), NULL, &n, NULL, &n, work, &lwork, &info);
    }
    delete[] work;
}

static void BM_dggev4(benchmark::State &state)
{
    int n = state.range(0);
    int r = state.range(1);
    double work_opt;
    int lwork = -1;
    int info;
    OpenBLASThreadContext __openblas_context(r);
    std::vector<int> jpvt(n);
    std::vector<double> alphar(n);
    std::vector<double> alphai(n);
    std::vector<double> beta(n);
    std::vector<double> A_start(n * n);
    std::vector<double> B_start(n * n);
    std::vector<double> A(n * n);
    std::vector<double> B(n * n);
    generate_random_matrix(A_start, 0);
    generate_random_matrix(B_start, 1);
    int ldA = n;
    int ldB = n;
    dggev4("N", "N", n, A.data(), ldA, B.data(), ldB, alphar.data(), alphai.data(), beta.data(), NULL, n, NULL, n, &work_opt, -1, jpvt.data(), &info);

    double *work = new double[(int)work_opt];
    lwork = (int)work_opt;
    for (auto _ : state) {
        memcpy(A.data(), A_start.data(), n * n * sizeof(double));
        memcpy(B.data(), B_start.data(), n * n * sizeof(double));
        memset(jpvt.data(), 0, n * sizeof(int));
        dggev4("N", "N", n, A.data(), ldA, B.data(), ldB, alphar.data(), alphai.data(), beta.data(), NULL, n, NULL, n, work, lwork, jpvt.data(), &info);
    }
    delete[] work;
}

static void BM_dgghd4(benchmark::State &state)
{
    const int n = state.range(0);
    const int r = state.range(1);
    double *A = (double *)malloc(n * n * sizeof(double));
    double *B = (double *)malloc(n * n * sizeof(double));
    double *Q = (double *)malloc(n * n * sizeof(double));
    double *Z = (double *)malloc(n * n * sizeof(double));
    double *A0 = (double *)malloc(n * n * sizeof(double));
    double *B0 = (double *)malloc(n * n * sizeof(double));
    double *work;
    double dummy[1];
    int info[1], ilo = 0, lwork = -1;

    OpenBLASThreadContext __openblas_context(r);

    generate_random_matrix(A0, n * n, 0);
    generate_random_matrix(B0, n * n, 1);
    for (int i = 0; i < n; i++) {
        B0[i * (n + 1)] = 2 * n;
        for (int j = i + 1; j < n; j++) {
            B0[i * n + j] = 0.0;
        }
    }

    dgghd4("I", "I", n, ilo, n, A, n, B, n, Q, n, Z, n, dummy, lwork, info);
    lwork = *dummy;
    work = (double *)malloc(lwork * sizeof(double));

    for (auto _ : state) {
        memcpy(A, A0, n * n * sizeof(double));
        memcpy(B, B0, n * n * sizeof(double));
        dgghd4("I", "I", n, ilo, n, A, n, B, n, Q, n, Z, n, work, lwork, info);
    }

    free(A);
    free(B);
    free(Q);
    free(Z);
    free(A0);
    free(B0);
    free(work);
}

static void BM_dgghd3(benchmark::State &state)
{
    const int n = state.range(0);
    const int r = state.range(1);
    double *A = (double *)malloc(n * n * sizeof(double));
    double *B = (double *)malloc(n * n * sizeof(double));
    double *Q = (double *)malloc(n * n * sizeof(double));
    double *Z = (double *)malloc(n * n * sizeof(double));
    double *A0 = (double *)malloc(n * n * sizeof(double));
    double *B0 = (double *)malloc(n * n * sizeof(double));
    double *work;
    double dummy[1];
    int info[1], ilo = 1, lwork = -1;

    OpenBLASThreadContext __openblas_context(r);

    generate_random_matrix(A0, n * n, 0);
    generate_random_matrix(B0, n * n, 1);
    for (int i = 0; i < n; i++) {
        B0[i * (n + 1)] = 2 * n;
        for (int j = i + 1; j < n; j++) {
            B0[i * n + j] = 0.0;
        }
    }

    dgghd3_("I", "I", &n, &ilo, &n, A, &n, B, &n, Q, &n, Z, &n, dummy, &lwork, info);
    lwork = *dummy;
    work = (double *)malloc(lwork * sizeof(double));

    for (auto _ : state) {
        memcpy(A, A0, n * n * sizeof(double));
        memcpy(B, B0, n * n * sizeof(double));
        dgghd3_("I", "I", &n, &ilo, &n, A, &n, B, &n, Q, &n, Z, &n, work, &lwork, info);
    }

    free(A);
    free(B);
    free(Q);
    free(Z);
    free(A0);
    free(B0);
    free(work);
}

static void BM_dggprp4(benchmark::State &state)
{
    const int n = state.range(0);
    const int r = state.range(1);
    double *A = (double *)malloc(n * n * sizeof(double));
    double *B = (double *)malloc(n * n * sizeof(double));
    double *Q = (double *)malloc(n * n * sizeof(double));
    double *Z = (double *)malloc(n * n * sizeof(double));
    double *A0 = (double *)malloc(n * n * sizeof(double));
    double *B0 = (double *)malloc(n * n * sizeof(double));
    int *jpvt = (int *)malloc(n * sizeof(int));
    double *work;
    double dummy[1];
    int info[1], lwork = -1;

    OpenBLASThreadContext __openblas_context(r);

    generate_random_matrix(A0, n * n, 0);
    generate_random_matrix(B0, n * n, 1);

    dggprp(1, 1, n, A, n, B, n, Q, n, Z, n, dummy, lwork, jpvt, info);
    lwork = *dummy;
    work = (double *)malloc(lwork * sizeof(double));

    for (auto _ : state) {
        memcpy(A, A0, n * n * sizeof(double));
        memcpy(B, B0, n * n * sizeof(double));
        dggprp(1, 1, n, A, n, B, n, Q, n, Z, n, work, lwork, jpvt, info);
    }

    free(A);
    free(B);
    free(Q);
    free(Z);
    free(A0);
    free(B0);
    free(work);
    free(jpvt);
}

static void BM_dggprp3(benchmark::State &state)
{
    const int n = state.range(0);
    const int r = state.range(1);
    double *A = (double *)malloc(n * n * sizeof(double));
    double *B = (double *)malloc(n * n * sizeof(double));
    double *Q = (double *)malloc(n * n * sizeof(double));
    double *Z = (double *)malloc(n * n * sizeof(double));
    double *A0 = (double *)malloc(n * n * sizeof(double));
    double *B0 = (double *)malloc(n * n * sizeof(double));
    int *jpvt = (int *)malloc(n * sizeof(int));
    double *work;
    double dummy[1];
    int info[1], lwork = -1;

    OpenBLASThreadContext __openblas_context(r);

    generate_random_matrix(A0, n * n, 0);
    generate_random_matrix(B0, n * n, 1);

    dggprp(1, 1, n, A, n, B, n, Q, n, Z, n, dummy, lwork, jpvt, info);
    lwork = *dummy;
    work = (double *)malloc(lwork * sizeof(double));

    for (auto _ : state) {
        memcpy(A, A0, n * n * sizeof(double));
        memcpy(B, B0, n * n * sizeof(double));
        // factor B
        dgeqrf(n, n, B, n, work, work + n, lwork, info);
        // apply Q.T to A
        dormqr("L", "T", n, n, n, B, n, work, A, n, work + n, lwork, info);
        // form Q
        memcpy(Q, B, n * n * sizeof(double));
        dorgqr(n, n, n, Q, n, work, work + n, lwork, info);
    }

    free(A);
    free(B);
    free(Q);
    free(Z);
    free(A0);
    free(B0);
    free(work);
    free(jpvt);
}

// --- Generalized Eigenvalue Problem Benchmarks ---
static void BM_dtgevc(benchmark::State &state)
{
    int n = state.range(0);
    int r = state.range(1);
    OpenBLASThreadContext __openblas_context(r);
    char side = 'B', howmny = 'B';
    double total_flops = (8.0 * n * n * n / 3.0) + (3 * n * n) - (11.0 * n / 3.0);

    std::vector<double> S(n * n), P(n * n);
    std::vector<double> alphar(n), alphai(n), beta(n);
    generate_generalized_real_triangular(n, S, P, alphar, alphai, beta);

    std::vector<double> VL{ eye(n) }, VR{ eye(n) };
    std::vector<double> work(std::max(1, 6 * n), 0.0);
    int m_out = 0, info = 0;

    for (auto _ : state) {
        dtgevc_(&side, &howmny, nullptr, &n, S.data(), &n, P.data(), &n, VL.data(), &n, VR.data(), &n, &n, &m_out, work.data(), &info);
    }

    state.SetComplexityN(n);
    state.counters["FLOPS"] = benchmark::Counter(static_cast<double>(state.iterations()) * total_flops, benchmark::Counter::kIsRate);
}

static void BM_dtgevc3(benchmark::State &state)
{
    int n = state.range(0);
    int r = state.range(1);
    OpenBLASThreadContext __openblas_context(r);
    char side = 'B', howmny = 'B';
    double total_flops = (8.0 * n * n * n / 3.0) + (3 * n * n) - (11.0 * n / 3.0);

    std::vector<double> S(n * n), P(n * n);
    std::vector<double> alphar(n), alphai(n), beta(n);
    generate_generalized_real_triangular(n, S, P, alphar, alphai, beta);

    std::vector<double> VL{ eye(n) }, VR{ eye(n) };
    int m_out = 0, info = 0;

    double dummy_work;
    dtgevc3(side, howmny, nullptr, n, S.data(), n, P.data(), n, alphar.data(), alphai.data(), beta.data(), VL.data(), n, VR.data(), n, n, &m_out, &dummy_work, -1, &info);

    int lwork = static_cast<int>(dummy_work);
    std::vector<double> work(lwork, 0.0);

    for (auto _ : state) {
        dtgevc3(side, howmny, nullptr, n, S.data(), n, P.data(), n, alphar.data(), alphai.data(), beta.data(), VL.data(), n, VR.data(), n, n, &m_out, work.data(), lwork, &info);
    }

    state.SetComplexityN(n);
    state.counters["FLOPS"] = benchmark::Counter(static_cast<double>(state.iterations()) * total_flops, benchmark::Counter::kIsRate);
}

// --- Standard Eigenvalue Problem Benchmarks ---
static void BM_dtrevc(benchmark::State &state)
{
    int n = state.range(0);
    int r = state.range(1);
    OpenBLASThreadContext __openblas_context(r);
    char side = 'B', howmny = 'B';
    double total_flops = (5.0 * n * n * n / 3.0) + (n * n) - (2.0 * n / 3.0);

    std::vector<double> T(n * n);
    generate_standard_real_triangular(n, T);

    std::vector<double> VL{ eye(n) }, VR{ eye(n) };
    std::vector<double> work(std::max(1, 3 * n), 0.0);
    int m_out = 0, info = 0;

    for (auto _ : state) {
        dtrevc_(&side, &howmny, nullptr, &n, T.data(), &n, VL.data(), &n, VR.data(), &n, &n, &m_out, work.data(), &info);
    }

    state.SetComplexityN(n);
    state.counters["FLOPS"] = benchmark::Counter(static_cast<double>(state.iterations()) * total_flops, benchmark::Counter::kIsRate);
}

static void BM_dtrevc3(benchmark::State &state)
{
    int n = state.range(0);
    int r = state.range(1);
    OpenBLASThreadContext __openblas_context(r);
    char side = 'B', howmny = 'B';
    double total_flops = (5.0 * n * n * n / 3.0) + (n * n) - (2.0 * n / 3.0);

    std::vector<double> T(n * n);
    generate_standard_real_triangular(n, T);

    std::vector<double> VL{ eye(n) }, VR{ eye(n) };
    int m_out = 0, info = 0;

    int lwork_query = -1;
    double dummy_work;
    dtrevc3_(&side, &howmny, nullptr, &n, T.data(), &n, VL.data(), &n, VR.data(), &n, &n, &m_out, &dummy_work, &lwork_query, &info);

    int lwork = static_cast<int>(dummy_work);
    lwork = n * 130;
    std::vector<double> work(lwork, 0.0);

    for (auto _ : state) {
        dtrevc3_(&side, &howmny, nullptr, &n, T.data(), &n, VL.data(), &n, VR.data(), &n, &n, &m_out, work.data(), &lwork, &info);
    }

    state.SetComplexityN(n);
    state.counters["FLOPS"] = benchmark::Counter(static_cast<double>(state.iterations()) * total_flops, benchmark::Counter::kIsRate);
}

static void apply_args(benchmark::internal::Benchmark *b)
{
    constexpr int num_threads[] = { 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24 };
    constexpr int sizes[] = { 8000 };
    for (const auto num : num_threads) {
        for (const auto size : sizes) {
            b->Args({ size, num });
        }
    }
    b->MinTime(2.0);
    b->Unit(benchmark::kMillisecond);
}

static void apply_args_evc(benchmark::internal::Benchmark *b)
{
    constexpr int num_threads[] = { 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24 };
    constexpr int sizes[] = { 16000 };
    for (const auto num : num_threads) {
        for (const auto size : sizes) {
            b->Args({ size, num });
        }
    }
    b->MinTime(2.0);
    b->Unit(benchmark::kMillisecond);
    b->Complexity();
}


BENCHMARK(BM_dggev3)->Apply(apply_args);
BENCHMARK(BM_dggev4)->Apply(apply_args);
BENCHMARK(BM_dgghd3)->Apply(apply_args);
BENCHMARK(BM_dgghd4)->Apply(apply_args);
BENCHMARK(BM_dggprp3)->Apply(apply_args);
BENCHMARK(BM_dggprp4)->Apply(apply_args);
BENCHMARK(BM_dtgevc)->Apply(apply_args_evc);
BENCHMARK(BM_dtgevc3)->Apply(apply_args_evc);
BENCHMARK(BM_dtrevc)->Apply(apply_args_evc);
BENCHMARK(BM_dtrevc3)->Apply(apply_args_evc);


BENCHMARK_MAIN();
