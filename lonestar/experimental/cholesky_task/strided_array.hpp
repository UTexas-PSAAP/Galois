#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

/* TODO: Now that the namespace is set up, shorten the class names. */
/*
 * TODO: Perhaps also move the aritmetic/comparison operators to
 * the surrounding namespace.
 */

namespace sa {

struct dim_data {
  std::size_t shape;
  std::ptrdiff_t stride;
  /*
   * Zero as default shape/stride should hopefully make uninitialized
   * use hang as fast as possible without corrupting data.
   */
  constexpr dim_data() noexcept : shape(0), stride(0) {}
  constexpr dim_data(const dim_data &) = default;
  constexpr dim_data(std::size_t sh, std::ptrdiff_t st) noexcept :
    shape(sh),
    stride(st) {}
  constexpr bool operator==(const dim_data &other) const noexcept {
    return shape == other.shape && stride == other.stride;
  }
  constexpr bool operator!=(const dim_data &other) const noexcept {
    return shape != other.shape || stride != other.stride;
  }
};

/* TODO: ssize_t is technically too small for the jump.
 * Perhaps the best thing would be to store the signs of the size_t entries
 * as an extra bitfield after the other struct members.
 * That'll probably require adding a bunch of special
 * overloads to handle that case though.
 * That's an issue with the array metadata layout too though.
 * If you have an array longer than the maximum ssize_t value,
 * how do you reverse it?
 * Offloading this to intptr_t may work, but ssize_t and intptr_t
 * are defined slightly differently. Do people use segmented architectures
 * anymore?
 */
struct slice {
  std::size_t start;
  std::size_t stop;
  std::ptrdiff_t jump;
  constexpr slice() noexcept :
    start(0),
    stop(std::numeric_limits<std::size_t>::max()),
    jump(1) {}
  explicit constexpr slice(std::size_t str) noexcept :
    start(str),
    stop(std::numeric_limits<std::size_t>::max()),
    jump(1) {}
  constexpr slice(std::size_t str, std::size_t stp) noexcept :
    start(str),
    stop(stp),
    jump(1) {}
  constexpr slice(std::size_t str,
                  std::size_t stp,
                  std::ptrdiff_t jmp) noexcept :
    start(str),
    stop(stp),
    jump(jmp) {}
};

namespace detail {
/** Some utilities to help with indexing: **/

/*
 * Get the cumulative number of non-integer dimensions passed after
 * a given index in the input parameter pack.
 */
template <typename... T>
constexpr decltype(auto) dimension_offsets() noexcept {
  std::size_t count = 0, total = 0;
  ((std::is_integral<T>::value ? total : total++), ...);

  /*
   * The evaluation order inside the braces is well-defined in the standard,
   * so we can do this in one expression, incrementing the count as we go.
   */
  std::array<std::size_t, sizeof...(T)> counts{
    std::is_integral<T>::value ? std::min(count, total - 1) : count++...};
  return counts;
}

inline constexpr std::ptrdiff_t axis_offset(dim_data axis,
                                            std::size_t index) noexcept {
  /*
   * If the compiler complains about a non-constexpr function in
   * this assertion, it's probably because there was a
   * compile-time out of bounds index.
   */
  assert(index < axis.shape); // Index out of bounds.
  return axis.stride * index;
}

inline constexpr std::ptrdiff_t axis_offset(dim_data axis,
                                            slice index) noexcept {
  return std::min(axis.shape, index.start) * axis.stride;
}

template <typename... T, size_t... pack_indices>
constexpr std::ptrdiff_t
get_offset(std::index_sequence<pack_indices...>,
           const std::array<dim_data, sizeof...(pack_indices)> axes,
           T... indices) noexcept {
  std::ptrdiff_t offset = 0;
  (offset += ... += axis_offset(axes[pack_indices], indices));
  return offset;
}

inline constexpr dim_data slice_axis(dim_data axis, slice index) noexcept {
  if (index.jump >= 0) {
    std::size_t real_stop = std::min(axis.shape, index.stop);
    if (index.start < real_stop) {
      std::size_t new_shape = (real_stop - index.start) / index.jump;
      assert(index.jump != 0 || new_shape <= 1);
      return dim_data(new_shape, axis.stride * index.jump);
    } else {
      return dim_data(0, 0);
    }
  } else {
    /* TODO: Is this really the right thing to do here? */
    std::size_t real_start = std::min(axis.shape, index.start);
    if (index.stop < real_start) {
      std::size_t new_shape = (real_start - index.stop) / index.jump;
      return dim_data(new_shape, axis.stride * index.jump);
    } else {
      return dim_data(0, 0);
    }
  }
}

inline constexpr dim_data slice_axis(dim_data axis,
                                     std::size_t index) noexcept {
  /*
   * Ideally this will never be called.
   * Unfortunately, putting an assert(false) here
   * upsets clang, so this will have to be good enough.
   */
  return dim_data(0, 0);
}

template <typename... T, std::size_t... pack_indices>
inline constexpr decltype(auto)
slice_axes(std::index_sequence<pack_indices...>,
           const std::array<dim_data, sizeof...(pack_indices)> axes,
           T... indices) noexcept {
  static_assert(sizeof...(pack_indices) == sizeof...(T),
                "Mismatch in input parameter pack sizes.");
  constexpr auto dim_offsets = dimension_offsets<T...>();
  std::array<dim_data, dim_offsets[sizeof...(T) - 1] + 1> new_axes;
  ((new_axes[dim_offsets[pack_indices]] =
      std::is_integral<T>::value ? new_axes[dim_offsets[pack_indices]]
                                 : slice_axis(axes[pack_indices], indices)),
   ...);
  return new_axes;
}

template <std::size_t dims, std::size_t... I>
inline constexpr decltype(auto)
transpose(std::index_sequence<I...>,
          const std::array<dim_data, dims> axes) noexcept {
  static_assert(sizeof...(I) + 2 == dims,
                "Mismatch between index sequence and axes provided.");
  return std::array<dim_data, dims>(
    {axes[I]..., axes[dims - 1], axes[dims - 2]});
}

// This is suitable for expansion in a fold expression.
// Just using the macro assert makes the compiler complain
// when building in release mode, so we instead have to
// rely on the optimizer to eliminate the dead code
// resulting from this function.
// Note: this has to be constexpr because it can be called
// inside a constexpr context, even though it won't actually
// return anything.
// Note: If the compiler complains about a call to a non-constexpr
// function inside the assert, that indicates that the assertion
// failure was triggered, but at compile time.
template <typename T>
static inline constexpr bool non_macro_assert(T &&args) noexcept {
  assert(std::forward<T>(args));
  return false;
}

template <std::size_t dims, std::size_t... I, typename... T>
inline constexpr decltype(auto)
check_blockable(std::index_sequence<I...>,
                const std::array<dim_data, dims> axes,
                T... block_sizes) noexcept {
  static_assert(dims == sizeof...(I));
  static_assert(dims == sizeof...(T));
  static_assert((std::is_same<std::size_t, T>::value && ...));
  (non_macro_assert(axes[I].shape % block_sizes == 0), ...);
}

template <std::size_t dims, std::size_t... I, typename... T>
inline constexpr decltype(auto)
block_axes(std::index_sequence<I...>,
           const std::array<dim_data, dims> axes,
           T... block_sizes) noexcept {
  return std::array<dim_data, 2 * dims>(
    {dim_data(axes[I].shape / block_sizes, axes[I].stride * block_sizes)...,
     dim_data(block_sizes, axes[I].stride)...});
}

template <std::size_t dims, std::size_t... i>
inline constexpr bool axes_equal(std::index_sequence<i...>,
                                 std::array<dim_data, dims> l,
                                 std::array<dim_data, dims> r) noexcept {
  return (true && ... && (l[i] == r[i]));
}

} // namespace detail

/* TODO: Check that only size_t indices are passed into the helper routines. */
/* TODO: Check size_t vs ssize_t everywhere. */
/* TODO: Check that as much as possible of this is noexcept. */
/* TODO: Prevent most of the symbols here from exporting from any dll/so. */
/* TODO: Mark most function/method arguments here as const. */
/* TODO: Check that negative strides are actually handled correctly. Test this.
 */
/* TODO: Make sure that stride arithmetic is always applied to void* types when
 * being mapped back to pointers. */
/* TODO: Maybe... route calls from strided_array methods directly into the
 * corresponding helper methods to reduce template instantiations. */

template <std::size_t dims>
struct array_axes {
  std::array<dim_data, dims> axes;
  constexpr array_axes(const array_axes &) = default;
  constexpr array_axes(array_axes &&) = default;
  constexpr array_axes(std::array<dim_data, dims> &&a) noexcept : axes(a) {}

  constexpr array_axes<dims> &operator=(const array_axes<dims> &) = default;

  template <std::size_t other_dims>
  constexpr bool operator==(const array_axes<other_dims> &other) const
    noexcept {
    if constexpr (other_dims != dims) {
      return false;
    } else {
      return detail::axes_equal(std::make_index_sequence<dims>(), axes,
                                other.axes);
    }
  }

  template <std::size_t other_dims>
  constexpr bool operator!=(const array_axes<other_dims> &other) const
    noexcept {
    return !(*this == other);
  }

  constexpr decltype(auto) operator[](std::size_t idx) const noexcept {
    // Assert no out of bounds access.
    assert(idx < axes.size());
    return axes[idx];
  }

  template <typename... T>
  constexpr std::ptrdiff_t offset(T... indices) const noexcept {
    return detail::get_offset(std::make_index_sequence<sizeof...(T)>(), axes,
                              indices...);
  }

  template <typename T>
  constexpr bool is_contiguous(size_t i) const noexcept {
    assert(i < dims);
    return axes[i].shape <= 1 || axes[i].stride == sizeof(T);
  }

  template <typename... T>
  constexpr decltype(auto) operator()(T... indices) const noexcept {
    auto new_axes = detail::slice_axes(
      std::make_index_sequence<sizeof...(T)>(), axes, indices...);
    return array_axes<new_axes.size()>(std::move(new_axes));
  }

  constexpr decltype(auto) transpose() const noexcept {
    static_assert(dims >= 2, "Cannot transpose 1D array.");
    return array_axes<dims>(
      detail::transpose(std::make_index_sequence<dims - 2>(), axes));
  }

  template <typename... I>
  constexpr decltype(auto) block(I... block_sizes) const noexcept {
    static_assert(sizeof...(I) == dims,
                  "Blocking only some axes is not implemented.");
    static_assert((true && ... && std::is_integral<I>::value),
                  "Non-integral type passed as block size.");
    /* Only positive block sizes are allowed. */
    (detail::non_macro_assert(block_sizes > 0), ...);
    detail::check_blockable(std::make_index_sequence<sizeof...(I)>(), axes,
                            std::size_t(block_sizes)...);
    return array_axes<2 * dims>(detail::block_axes(
      std::make_index_sequence<sizeof...(I)>(), axes, block_sizes...));
  }
};

namespace detail {

template <typename T, size_t dims, size_t... i>
inline void assert_aligned_strides(T *,
                                   array_axes<dims> axes,
                                   std::index_sequence<i...>) noexcept {
  (detail::non_macro_assert(axes[i].shape == 1 || axes[i].stride % std::alignment_of_v<T> == 0), ...);
}

} // namespace detail

/*
 * A struct carrying the needed metadata for a strided array.
 * This one's really only intended to provide a minimal indexing
 * and reshaping interface. It doesn't do any resource management.
 */
template <typename T, std::size_t dims>
struct strided_array {

  static_assert(dims > 0, "Zero-dimensional arrays are not allowed.");

  T * data;
  array_axes<dims> axes;
  constexpr strided_array() = default;

  constexpr strided_array(T *p, array_axes<dims> ax) noexcept :
    data(p),
    axes(ax) {
    detail::assert_aligned_strides(data, axes,
                                   std::make_index_sequence<dims>());
  }

  constexpr strided_array<T, dims> &operator=(const strided_array<T, dims> &) = default;

  bool is_contiguous(std::size_t i) const noexcept {
    return axes.template is_contiguous<T>(i);
  }

  decltype(auto) transpose() const noexcept {
    return strided_array<T, dims>(data, axes.transpose());
  }

  template <typename... I>
  decltype(auto) block(I... i) const noexcept {
    return strided_array<T, dims + sizeof...(I)>(data, axes.block(i...));
  }

  template <typename... I>
  decltype(auto) operator()(I... indices) const noexcept {
    /* TODO: Relax this constraint. */
    static_assert(
      sizeof...(I) == dims,
      "The number of indices provided does not match the array dimensions.");
    std::size_t offset = axes.offset(indices...);
    if constexpr ((true && ... && std::is_integral<I>::value)) {
      /* All indices are integers, so return the corresponding array entry. */
      return *((T *)((char *)data + offset));
    } else {
      /* Slice the array. */
      auto new_axes = axes(indices...);
      return strided_array<T, new_axes.axes.size()>(
        (T *)((char *)data + offset), new_axes);
    }
  }
};

} // namespace sa
