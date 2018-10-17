#pragma once

#include <complex>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "strided_array.hpp"

namespace sa {

template <typename... T>
void debug_print(T... i) noexcept {
  (std::cout << ... << i) << std::endl;
}

namespace cublas {

/*
 * TODO: Do the cublas routines support negative ld* values?
 * If not, assert that the input strides are positive in the wrappers.
 * What about zero length dimensions?
 */

/*
 * While it seems like this would go without saying, perhaps it would
 * be worth static asserting that T is trivially copyable?
 * Alternatively, since cublas only works on specific types,
 * just check that T is one of those types.
 */

namespace detail {

template <typename T>
constexpr int get_axis_ld(dim_data d) noexcept {
  // No overflow when converting stride to int to match BLAS interface.
  assert(std::numeric_limits<int>::min() <= d.stride &&
         d.stride <= std::numeric_limits<int>::max());
  // Also make sure that the stride is a multiple of sizeof(T).
  assert(d.stride % sizeof(T) == 0 || d.shape <= 1);
  return d.shape > 1 ? d.stride / sizeof(T) : 1;
}

} // namespace detail

// cublas requires initialization.
// This is a RAII wrapper around it.
// It can be moved, but not copied.
// TODO: borrowing?
struct context {

  cublasHandle_t handle;

  context() noexcept {
    auto status = cublasCreate(&handle);
    assert(status == CUBLAS_STATUS_SUCCESS);
  }

  context(const context &) = delete;

  explicit context(cublasHandle_t h) noexcept : handle(h) {}

  context(context &&other) noexcept {
    // Assert that other hasn't already been moved from.
    assert(other.handle != nullptr);

    handle = other.handle;
    other.handle = nullptr;
  }

  context &operator=(const context &) = delete;

  context &operator=(context &&other) noexcept {
    // Assert that the other hasn't already been moved from.
    assert(other.handle != nullptr);
    /*
     * Also check that this one hasn't already been initialized.
     * TODO: Currently this just disallows move assignment in to
     * something that has already been initialized. Should something
     * like that actually be allowed, and, if so, what should it do?
     */
    assert(handle == nullptr);

    handle = other.handle;
    other.handle = nullptr;
    return *this;
  }

  ~context() noexcept {
    if (handle != nullptr) {
      auto stat = cublasDestroy(handle);
      assert(stat == CUBLAS_STATUS_SUCCESS);
    }
  }

  /*
   * TODO: Make the various functions requiring this context callable
   * as methods of the context class (think UFCS).
   */
};

/* TODO: Add shape-only comparison to check for compatible dimensions. */

template <typename T>
cublasStatus_t SetMatrix(strided_array<T, 2> A,
                         strided_array<T, 2> B) noexcept {
  // For a copy to work, the dimensions must match.
  assert(A.axes[0].shape == B.axes[0].shape);
  assert(A.axes[1].shape == B.axes[1].shape);

  // Array dimensions must not overflow when cast to int
  // so they can be passed to the BLAS interface.
  assert(A.axes[0].shape < std::numeric_limits<int>::max());
  assert(A.axes[1].shape < std::numeric_limits<int>::max());

  if (A.is_contiguous(0)) {
    // If A has this data layout, B must have this layout too.
    // This routine performs a copy, not a transpose.
    assert(B.is_contiguous(0));
    int lda = detail::get_axis_ld<T>(A.axes[1]),
        ldb = detail::get_axis_ld<T>(B.axes[1]);
    return cublasSetMatrix(A.axes[0].shape, A.axes[1].shape, sizeof(T),
                           reinterpret_cast<void *>(A.data), lda,
                           reinterpret_cast<void *>(B.data), ldb);
  } else {
    // A must be contiguous along at least one axis.
    assert(A.is_contiguous(1));
    // As before, if A has this data layout, B must as well.
    assert(B.is_contiguous(1));
    int lda = detail::get_axis_ld<T>(A.axes[0]),
        ldb = detail::get_axis_ld<T>(B.axes[0]);
    return cublasSetMatrix(A.axes[1].shape, A.axes[0].shape, sizeof(T),
                           reinterpret_cast<void *>(A.data), lda,
                           reinterpret_cast<void *>(B.data), ldb);
  }
}

template <typename T>
cublasStatus_t SetMatrixAsync(strided_array<T, 2> A,
                              strided_array<T, 2> B,
                              cudaStream_t stream) noexcept {
  // For a copy to work, the dimensions must match.
  assert(A.axes[0].shape == B.axes[0].shape);
  assert(A.axes[1].shape == B.axes[1].shape);

  // Array dimensions must not overflow when cast to int
  // so they can be passed to the BLAS interface.
  assert(A.axes[0].shape < std::numeric_limits<int>::max());
  assert(A.axes[1].shape < std::numeric_limits<int>::max());

  if (A.is_contiguous(0)) {
    // If A has this data layout, B must have this layout too.
    // This routine performs a copy, not a transpose.
    assert(B.is_contiguous(0));
    int lda = detail::get_axis_ld<T>(A.axes[1]),
        ldb = detail::get_axis_ld<T>(B.axes[1]);
    return cublasSetMatrixAsync(A.axes[0].shape, A.axes[1].shape, sizeof(T),
                                reinterpret_cast<void *>(A.data), lda,
                                reinterpret_cast<void *>(B.data), ldb, stream);
  } else {
    // A must be contiguous along at least one axis.
    assert(A.is_contiguous(1));
    // As before, if A has this data layout, B must as well.
    assert(B.is_contiguous(1));
    int lda = detail::get_axis_ld<T>(A.axes[0]),
        ldb = detail::get_axis_ld<T>(B.axes[0]);
    return cublasSetMatrixAsync(A.axes[1].shape, A.axes[0].shape, sizeof(T),
                                reinterpret_cast<void *>(A.data), lda,
                                reinterpret_cast<void *>(B.data), ldb, stream);
  }
}

template <typename T>
cublasStatus_t GetMatrix(strided_array<T, 2> A,
                         strided_array<T, 2> B) noexcept {
  // Dimensions of A and B must match for a copy.
  assert(A.axes[0].shape == B.axes[0].shape);
  assert(A.axes[1].shape == B.axes[1].shape);

  // Array dimensions must not overflow when cast to int
  // so they can be passed to the BLAS interface.
  assert(A.axes[0].shape < std::numeric_limits<int>::max());
  assert(A.axes[1].shape < std::numeric_limits<int>::max());

  if (A.is_contiguous(0)) {
    // If A has this layout, B must too.
    assert(B.is_contiguous(0));
    int lda = detail::get_axis_ld<T>(A.axes[1]),
        ldb = detail::get_axis_ld<T>(B.axes[1]);
    return cublasGetMatrix(A.axes[0].shape, A.axes[1].shape, sizeof(T),
                           reinterpret_cast<void *>(A.data), lda,
                           reinterpret_cast<void *>(B.data), ldb);
  } else {
    // A must be contiguous along at least one axis.
    assert(A.is_contiguous(1));
    // If A has this layout, B must too.
    assert(B.is_contiguous(1));
    int lda = detail::get_axis_ld<T>(A.axes[0]),
        ldb = detail::get_axis_ld<T>(B.axes[0]);
    return cublasGetMatrix(A.axes[1].shape, A.axes[0].shape, sizeof(T),
                           reinterpret_cast<void *>(A.data), lda,
                           reinterpret_cast<void *>(B.data), ldb);
  }
}

template <typename T>
cublasStatus_t GetMatrixAsync(strided_array<T, 2> A,
                              strided_array<T, 2> B,
                              cudaStream_t stream) noexcept {
  // Dimensions of A and B must match for a copy.
  assert(A.axes[0].shape == B.axes[0].shape);
  assert(A.axes[1].shape == B.axes[1].shape);

  // Array dimensions must not overflow when cast to int
  // so they can be passed to the BLAS interface.
  assert(A.axes[0].shape < std::numeric_limits<int>::max());
  assert(A.axes[1].shape < std::numeric_limits<int>::max());

  if (A.is_contiguous(0)) {
    // If A has this layout, B must too.
    assert(B.is_contiguous(0));
    int lda = detail::get_axis_ld<T>(A.axes[1]),
        ldb = detail::get_axis_ld<T>(B.axes[1]);
    return cublasGetMatrixAsync(A.axes[0].shape, A.axes[1].shape, sizeof(T),
                                reinterpret_cast<void *>(A.data), lda,
                                reinterpret_cast<void *>(B.data), ldb, stream);
  } else {
    // A must be contiguous along at least one axis.
    assert(A.is_contiguous(1));
    // If A has this layout, B must too.
    assert(B.is_contiguous(1));
    int lda = detail::get_axis_ld<T>(A.axes[0]),
        ldb = detail::get_axis_ld<T>(B.axes[0]);
    return cublasGetMatrixAsync(A.axes[1].shape, A.axes[0].shape, sizeof(T),
                                reinterpret_cast<void *>(A.data), lda,
                                reinterpret_cast<void *>(B.data), ldb, stream);
  }
}

namespace detail {

inline cublasFillMode_t invert_fill_mode(cublasFillMode_t f) noexcept {
  return f == CUBLAS_FILL_MODE_LOWER ? CUBLAS_FILL_MODE_UPPER
                                     : CUBLAS_FILL_MODE_LOWER;
}

inline cublasSideMode_t invert_side_mode(cublasSideMode_t s) noexcept {
  return s == CUBLAS_SIDE_LEFT ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT;
}

inline cublasOperation_t invert_operation_mode(cublasOperation_t o) noexcept {
  /*
   * Note: in the BLAS interface, the 'c' parameter means the same thing as
   * 't'. At least the documentation/implementation of dtrsm implies that. It
   * appears that this isn't the case for the cublas interface though.
   */
  /*
   * TODO: Confirm what the cublas behavior is for conjugate transpose and
   * determine what to do about it. For now, don't mess with conjugate
   * transposes though.
   */
  assert(o != CUBLAS_OP_C);
  return o == CUBLAS_OP_N ? CUBLAS_OP_T : CUBLAS_OP_N;
}

} // namespace detail

/*
 * TODO: Should we follow move/return semantics for managing the context
 * for the wrapped routines that require it?
 */

// template <typename T>
// cublasStatus_t trsm(context &c, cublasSideMode_t side, cublasFillMode_t,
// cublasOperation_t trans, cublasDiagType_t diag, T alpha, strided_array<T, 2>
// A, strided_array<T, 2> B) noexcept {
//  /*
//   * A and B are both required to be contiguous along at least one dimension,
//   * but there are different ways to call trsm to get the desired result
//   depending on
//   * whether A is transposed or not, whether the upper or lower part of A is
//   used,
//   * and which side of the left hand side A appears on in the equation to be
//   solved.
//   */
//  /* TODO: Implement something other than the both column contiguous case. */
//  if (A.axes.axes[0].shape == 1 || A.axes.axes[0].stride == sizeof(T)) {
//  }
//}

namespace detail {

template <typename T>
cublasStatus_t gemm(cublasHandle_t handle,
                    cublasOperation_t transa,
                    cublasOperation_t transb,
                    int m,
                    int n,
                    int k,
                    T alpha,
                    const T *A,
                    int lda,
                    const T *B,
                    int ldb,
                    T beta,
                    T *C,
                    int ldc) noexcept {
  // The point of defining this overload at all is to
  // provide a more meaningful error message.
  // The condition here will always be false when this
  // version is actually instantiated, but if you just
  // leave it as "false", some compilers eagerly evaluate
  // the static assertion and fail erroneously.
  static_assert(std::is_same<T, double>::value ||
                  std::is_same<T, double>::value ||
                  std::is_same<T, std::complex<float>>::value ||
                  std::is_same<T, std::complex<double>>::value,
                "Only, float, double, float complex, and double "
                "complex types are supported.");
  // Now return something to pacify the compiler warnings.
  return CUBLAS_STATUS_SUCCESS;
}

template <>
cublasStatus_t gemm(cublasHandle_t handle,
                    cublasOperation_t transa,
                    cublasOperation_t transb,
                    int m,
                    int n,
                    int k,
                    float alpha,
                    const float *A,
                    int lda,
                    const float *B,
                    int ldb,
                    float beta,
                    float *C,
                    int ldc) noexcept {
  return cublasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb,
                     &beta, C, ldc);
}

template <>
cublasStatus_t gemm(cublasHandle_t handle,
                    cublasOperation_t transa,
                    cublasOperation_t transb,
                    int m,
                    int n,
                    int k,
                    double alpha,
                    const double *A,
                    int lda,
                    const double *B,
                    int ldb,
                    double beta,
                    double *C,
                    int ldc) noexcept {
  return cublasDgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb,
                     &beta, C, ldc);
}

template <>
cublasStatus_t gemm(cublasHandle_t handle,
                    cublasOperation_t transa,
                    cublasOperation_t transb,
                    int m,
                    int n,
                    int k,
                    std::complex<float> alpha,
                    const std::complex<float> *A,
                    int lda,
                    const std::complex<float> *B,
                    int ldb,
                    std::complex<float> beta,
                    std::complex<float> *C,
                    int ldc) noexcept {
  return cublasCgemm(handle, transa, transb, m, n, k,
                     reinterpret_cast<const cuComplex *>(&alpha),
                     reinterpret_cast<const cuComplex *>(A), lda,
                     reinterpret_cast<const cuComplex *>(B), ldb,
                     reinterpret_cast<const cuComplex *>(&beta),
                     reinterpret_cast<cuComplex *>(C), ldc);
}

cublasStatus_t gemm(cublasHandle_t handle,
                    cublasOperation_t transa,
                    cublasOperation_t transb,
                    int m,
                    int n,
                    int k,
                    std::complex<double> alpha,
                    const std::complex<double> *A,
                    int lda,
                    const std::complex<double> *B,
                    int ldb,
                    std::complex<double> beta,
                    std::complex<double> *C,
                    int ldc) noexcept {
  return cublasZgemm(handle, transa, transb, m, n, k,
                     reinterpret_cast<const cuDoubleComplex *>(&alpha),
                     reinterpret_cast<const cuDoubleComplex *>(A), lda,
                     reinterpret_cast<const cuDoubleComplex *>(B), ldb,
                     reinterpret_cast<const cuDoubleComplex *>(&beta),
                     reinterpret_cast<cuDoubleComplex *>(C), ldc);
}

template <typename T>
cublasStatus_t trsm(cublasHandle_t handle,
                    cublasSideMode_t side,
                    cublasFillMode_t uplo,
                    cublasOperation_t trans,
                    cublasDiagType_t diag,
                    int m,
                    int n,
                    T alpha,
                    const T *A,
                    int lda,
                    T *B,
                    int ldb) noexcept {
  // The point of defining this overload at all is to
  // provide a more meaningful error message.
  // The condition here will always be false when this
  // version is actually instantiated, but if you just
  // leave it as "false", some compilers eagerly evaluate
  // the static assertion and fail erroneously.
  static_assert(std::is_same<T, double>::value ||
                  std::is_same<T, double>::value ||
                  std::is_same<T, std::complex<float>>::value ||
                  std::is_same<T, std::complex<double>>::value,
                "Only, float, double, float complex, and double "
                "complex types are supported.");
  // Now return something to pacify the compiler warnings.
  return CUBLAS_STATUS_SUCCESS;
}

template <>
cublasStatus_t trsm<float>(cublasHandle_t handle,
                           cublasSideMode_t side,
                           cublasFillMode_t uplo,
                           cublasOperation_t trans,
                           cublasDiagType_t diag,
                           int m,
                           int n,
                           float alpha,
                           const float *A,
                           int lda,
                           float *B,
                           int ldb) noexcept {
  return cublasStrsm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B,
                     ldb);
}

template <>
cublasStatus_t trsm<double>(cublasHandle_t handle,
                            cublasSideMode_t side,
                            cublasFillMode_t uplo,
                            cublasOperation_t trans,
                            cublasDiagType_t diag,
                            int m,
                            int n,
                            double alpha,
                            const double *A,
                            int lda,
                            double *B,
                            int ldb) noexcept {
  return cublasDtrsm(handle, side, uplo, trans, diag, m, n, &alpha, A, lda, B,
                     ldb);
}

template <>
cublasStatus_t trsm<std::complex<float>>(cublasHandle_t handle,
                                         cublasSideMode_t side,
                                         cublasFillMode_t uplo,
                                         cublasOperation_t trans,
                                         cublasDiagType_t diag,
                                         int m,
                                         int n,
                                         std::complex<float> alpha,
                                         const std::complex<float> *A,
                                         int lda,
                                         std::complex<float> *B,
                                         int ldb) noexcept {
  return cublasCtrsm(handle, side, uplo, trans, diag, m, n,
                     reinterpret_cast<const cuComplex *>(&alpha),
                     reinterpret_cast<const cuComplex *>(A), lda,
                     reinterpret_cast<cuComplex *>(B), ldb);
}

template <>
cublasStatus_t trsm<std::complex<double>>(cublasHandle_t handle,
                                          cublasSideMode_t side,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag,
                                          int m,
                                          int n,
                                          std::complex<double> alpha,
                                          const std::complex<double> *A,
                                          int lda,
                                          std::complex<double> *B,
                                          int ldb) noexcept {
  return cublasZtrsm(handle, side, uplo, trans, diag, m, n,
                     reinterpret_cast<const cuDoubleComplex *>(&alpha),
                     reinterpret_cast<const cuDoubleComplex *>(A), lda,
                     reinterpret_cast<cuDoubleComplex *>(B), ldb);
}

} // namespace detail

template <typename T>
cublasStatus_t gemm(context &ctx,
                    T alpha,
                    strided_array<T, 2> A,
                    strided_array<T, 2> B,
                    T beta,
                    strided_array<T, 2> C) noexcept {
  /*
   * A, B, and C are all required to be contiguous along at least one
   * dimension, but there are different ways to call gemm to get the desired
   * result depending on whether each array is contiguous along its rows or
   * columns.
   */

  // Check that no overflow will occur with the change in integer size
  // between std::size_t and the standard BLAS interface usage of int.
  assert(A.axes[0].shape <= std::numeric_limits<int>::max());
  assert(A.axes[1].shape <= std::numeric_limits<int>::max());
  assert(B.axes[0].shape <= std::numeric_limits<int>::max());
  assert(B.axes[1].shape <= std::numeric_limits<int>::max());
  assert(C.axes[0].shape <= std::numeric_limits<int>::max());
  assert(C.axes[1].shape <= std::numeric_limits<int>::max());

  // Check that the axes of the arrays all match.
  assert(A.axes[1].shape == B.axes[0].shape);
  assert(A.axes[0].shape == C.axes[0].shape);
  assert(B.axes[1].shape == B.axes[1].shape);

  int m = A.axes[0].shape, n = B.axes[1].shape, k = A.axes[1].shape;
  cublasOperation_t transa, transb;
  int lda, ldb, ldc;

  if (C.is_contiguous(1)) {
    ldc = detail::get_axis_ld<T>(C.axes[0]);

    if (A.is_contiguous(1)) {
      transa = CUBLAS_OP_N;
      lda = detail::get_axis_ld<T>(A.axes[0]);
    } else {
      // A must be contiguous along at least one axis.
      assert(A.is_contiguous(0));
      transa = CUBLAS_OP_T;
      lda = detail::get_axis_ld<T>(A.axes[1]);
    }

    if (B.is_contiguous(1)) {
      transb = CUBLAS_OP_N;
      ldb = detail::get_axis_ld<T>(B.axes[0]);
    } else {
      // B must be contiguous along at least one axis.
      assert(B.is_contiguous(0));
      transb = CUBLAS_OP_T;
      ldb = detail::get_axis_ld<T>(B.axes[1]);
    }

    // Swap order of A and B and use reversed transposes.
    // This is to handle C being row contiguous instead of column contiguous.
    // Mathematically, if $C = A B$, then $C^T = B^T A^T$.
    return detail::gemm(ctx.handle, transb, transa, n, m, k, alpha, B.data,
                        ldb, A.data, lda, beta, C.data, ldc);
  } else {
    // C must be contiguous along at least one axis.
    assert(C.is_contiguous(0));

    ldc = detail::get_axis_ld<T>(C.axes[1]);

    if (A.is_contiguous(0)) {
      transa = CUBLAS_OP_N;
      lda = detail::get_axis_ld<T>(A.axes[1]);
    } else {
      // A must be contiguous along at least one axis.
      assert(A.is_contiguous(1));
      transa = CUBLAS_OP_T;
      lda = detail::get_axis_ld<T>(A.axes[0]);
    }

    if (B.is_contiguous(0)) {
      transb = CUBLAS_OP_N;
      ldb = detail::get_axis_ld<T>(B.axes[1]);
    } else {
      // B must be contiguous along at least one axis.
      assert(B.is_contiguous(1));
      transb = CUBLAS_OP_T;
      ldb = detail::get_axis_ld<T>(B.axes[0]);
    }

    return detail::gemm(ctx.handle, transa, transb, m, n, k, alpha, A.data,
                        lda, B.data, ldb, beta, C.data, ldc);
  }
}

template <typename T>
cublasStatus_t trsm(context &ctx,
                    cublasSideMode_t side,
                    cublasFillMode_t uplo,
                    cublasDiagType_t diag,
                    T alpha,
                    strided_array<T, 2> A,
                    strided_array<T, 2> B) noexcept {
  cublasOperation_t trans;
  int lda, ldb, m, n;
  if (B.is_contiguous(1)) {
    n = B.axes[1].shape;
    m = B.axes[0].shape;
    if (side == CUBLAS_SIDE_LEFT) {
      assert(A.axes[0].shape == m);
      assert(A.axes[1].shape == m);
    } else {
      assert(A.axes[0].shape == n);
      assert(A.axes[1].shape == n);
    }
    ldb = detail::get_axis_ld<T>(B.axes[0]);
    side = detail::invert_side_mode(side);
    if (A.is_contiguous(1)) {
      lda = detail::get_axis_ld<T>(A.axes[0]);
      trans = CUBLAS_OP_N;
    } else {
      assert(A.is_contiguous(0));
      lda = detail::get_axis_ld<T>(A.axes[1]);
      trans = CUBLAS_OP_T;
      uplo = detail::invert_fill_mode(uplo);
    }
  } else {
    assert(B.is_contiguous(0));
    m = B.axes[0].shape;
    n = B.axes[1].shape;
    if (side == CUBLAS_SIDE_LEFT) {
      assert(A.axes[0].shape == m);
      assert(A.axes[1].shape == m);
    } else {
      assert(A.axes[0].shape == n);
      assert(A.axes[1].shape == n);
    }
    ldb = detail::get_axis_ld<T>(B.axes[1]);
    if (A.is_contiguous(0)) {
      lda = detail::get_axis_ld<T>(A.axes[1]);
      trans = CUBLAS_OP_N;
    } else {
      assert(A.is_contiguous(1));
      lda = detail::get_axis_ld<T>(A.axes[0]);
      trans = CUBLAS_OP_T;
      uplo = detail::invert_fill_mode(uplo);
    }
  }
  std::cout << m << " " << n << " " << alpha << " " << A.data << " " << lda
            << " " << B.data << " " << ldb << std::endl;
  return detail::trsm<T>(ctx.handle, side, uplo, trans, diag, m, n, alpha,
                         A.data, lda, B.data, ldb);
}

} // namespace cublas
} // namespace sa
