/**
 * Default constructor clearing the container on the host side.
 */
template<size_t size>
CudaMatrixContainer<size>::CudaMatrixContainer()
{
  // Data is cleared only on the host size
  #ifndef  __CUDA_ARCH__
    for (size_t i = 0; i < size; i++)
    {
      mMatrixContainer[i].floatData = nullptr;
    }
  #endif
}// end of default constructor.