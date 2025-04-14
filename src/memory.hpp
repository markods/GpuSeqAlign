#ifndef INCLUDE_MEMORY_HPP
#define INCLUDE_MEMORY_HPP

#include <cuda_runtime.h>
#include <memory>
#include <vector>

// create an uninitialized array on the host
template <typename T>
class HostArray
{
public:
    HostArray()
        : _arr {nullptr, [](T*) { }},
          _size {}
    { }

    void init(size_t size)
    {
        if (_size == size)
        {
            return;
        }

        T* pAlloc = nullptr;
        if (size > 0)
        {
            pAlloc = (T*)malloc(size * sizeof(T));
            if (pAlloc == nullptr)
            {
                throw std::bad_alloc();
            }
        }

        pointer arr {pAlloc, [](T* ptr)
        {
            if (ptr != nullptr)
            {
                free(ptr);
            }
        }};

        std::swap(_arr, arr);
        _size = size;
    }
    void clear()
    {
        init(0);
    }

    T& operator[](size_t pos)
    {
        return data()[pos];
    }
    const T& operator[](size_t pos) const
    {
        return data()[pos];
    }

    T* data()
    {
        return _arr.get();
    }
    const T* data() const
    {
        return _arr.get();
    }

    size_t size() const
    {
        return _size;
    }

private:
    using pointer = std::unique_ptr<T, void (*)(T*)>;
    pointer _arr;
    size_t _size;
};

// create an uninitialized array on the device
template <typename T>
class DeviceArray
{
public:
    DeviceArray()
        : _arr {nullptr, [](T*) { }},
          _size {}
    { }

    void init(size_t size)
    {
        if (_size == size)
        {
            return;
        }

        T* pAlloc = nullptr;
        if (size > 0)
        {
            if (cudaSuccess != cudaMalloc(&pAlloc, size * sizeof(T)))
            {
                throw std::bad_alloc();
            }
        }

        pointer arr {pAlloc, [](T* ptr)
        {
            if (ptr != nullptr)
            {
                cudaFree(ptr);
            }
        }};

        std::swap(_arr, arr);
        _size = size;
    }
    void clear()
    {
        init(0);
    }

    T* data()
    {
        return _arr.get();
    }
    const T* data() const
    {
        return _arr.get();
    }

    size_t size() const
    {
        return _size;
    }

private:
    using pointer = std::unique_ptr<T, void (*)(T*)>;
    pointer _arr;
    size_t _size;
};

// initialize memory on the device starting from the given element
template <typename T>
cudaError_t memSet(
    T* const arr,
    int idx,
    size_t count,
    int value)
{
    cudaError_t status = cudaMemset(
        /*devPtr*/ &arr[idx],       // Pointer to device memory.
        /*value*/ value,            // Value to set for each byte of specified memory.
        /*count*/ count * sizeof(T) // Size in bytes to set.
    );

    return status;
}
template <typename T>
cudaError_t memSet(
    DeviceArray<T>& arr,
    int idx,
    int value)
{
    return memSet(arr.data(), idx, arr.size() - idx, value);
}

// transfer data between the host and the device
template <typename T>
cudaError_t memTransfer(
    T* const dst,
    const T* const src,
    int elemcnt,
    cudaMemcpyKind kind)
{
    cudaError_t status = cudaMemcpy(
        /*dst*/ dst,                   // Destination memory address.
        /*src*/ src,                   // Source memory address.
        /*count*/ elemcnt * sizeof(T), // Size in bytes to copy.
        /*kind*/ kind                  // Type of transfer.
    );

    return status;
}
template <typename T>
cudaError_t memTransfer(
    DeviceArray<T>& dst,
    const std::vector<T>& src,
    int elemcnt)
{
    return memTransfer(dst.data(), src.data(), elemcnt, cudaMemcpyHostToDevice);
}
template <typename T>
cudaError_t memTransfer(
    HostArray<T>& dst,
    const DeviceArray<T>& src,
    int elemcnt)
{
    return memTransfer(dst.data(), src.data(), elemcnt, cudaMemcpyDeviceToHost);
}

// transfer a pitched matrix to a contiguous matrix, between the host and the device
// Dst and src cannot overlap
template <typename T>
cudaError_t memTransfer(
    T* const dst,
    const T* const src,
    int dst_rows,
    int dst_cols,
    int src_cols,
    cudaMemcpyKind kind)
{
    cudaError_t status = cudaMemcpy2D(
        /*dst*/ dst,                     // Destination memory address.
        /*dpitch*/ dst_cols * sizeof(T), // Pitch of destination memory (padded row size in bytes; distance between the starting points of two rows).
        /*src*/ src,                     // Source memory address.
        /*spitch*/ src_cols * sizeof(T), // Pitch of source memory (padded row size in bytes).

        /*width*/ dst_cols * sizeof(T), // Width of matrix transfer (non-padding row size in bytes).
        /*height*/ dst_rows,            // Height of matrix transfer (#rows).
        /*kind*/ kind                   // Type of transfer.
    );

    return status;
}
template <typename T>
cudaError_t memTransfer(
    HostArray<T>& dst,
    const DeviceArray<T>& src,
    int dst_rows,
    int dst_cols,
    int src_cols)
{
    return memTransfer(dst.data(), src.data(), dst_rows, dst_cols, src_cols, cudaMemcpyDeviceToHost);
}

#endif // INCLUDE_MEMORY_HPP
