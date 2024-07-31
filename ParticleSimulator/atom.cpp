#include "atom.h"
#include <iostream>
#include <mpi.h>

namespace ATOM {

template<class T>
atom<T>::atom(T x, T y, T z, T mass)
    : x(x), y(y), z(z), mass(mass), vx(0), vy(0), vz(0)
{
}

// Getters
template<class T>
T atom<T>::getX() const { return x; }

template<class T>
T atom<T>::getY() const { return y; }

template<class T>
T atom<T>::getZ() const { return z; }

template<class T>
T atom<T>::getMass() const { return mass; }

template<class T>
T atom<T>::getVx() const { return vx; }

template<class T>
T atom<T>::getVy() const { return vy; }

template<class T>
T atom<T>::getVz() const { return vz; }

// Setters
template<class T>
void atom<T>::setX(T x) { this->x = x; }

template<class T>
void atom<T>::setY(T y) { this->y = y; }

template<class T>
void atom<T>::setZ(T z) { this->z = z; }

template<class T>
void atom<T>::setMass(T mass) { this->mass = mass; }

template<class T>
void atom<T>::setVx(T vx) { this->vx = vx; }

template<class T>
void atom<T>::setVy(T vy) { this->vy = vy; }

template<class T>
void atom<T>::setVz(T vz) { this->vz = vz; }

// Print position
template<class T>
void atom<T>::printPosition() const {
    std::cout << "Position: (" << x << ", " << y << ", " << z << ")" << std::endl;
}

// MPI pack method
template<class T>
void atom<T>::pack(MPI_Comm comm, void *outbuf, int outsize, int *position) const {
    MPI_Pack(&x, 1, MPI_DOUBLE, outbuf, outsize, position, comm ); 
    MPI_Pack(&y, 1, MPI_DOUBLE, outbuf, outsize, position, comm ); 
    MPI_Pack(&z, 1, MPI_DOUBLE, outbuf, outsize, position, comm ); 
}

// MPI unpack method
template<class T>
void atom<T>::unpack(MPI_Comm comm, void *inbuf, int insize, int *position) {
    MPI_Unpack(inbuf, insize, position, &x, 1, MPI_DOUBLE, comm);
    MPI_Unpack(inbuf, insize, position, &y, 1, MPI_DOUBLE, comm);
    MPI_Unpack(inbuf, insize, position, &z, 1, MPI_DOUBLE, comm);
}


// Calculate buffer size needed for packing
template<class T>
int atom<T>::bufferSize(MPI_Comm comm) {
    int position = 0;
    MPI_Pack_size(1, MPI_DOUBLE, comm, &position);
    int size = 3 * position; // 7 fields of type double
    return size;
}

} // namespace ATOM
