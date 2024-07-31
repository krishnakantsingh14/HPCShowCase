#ifndef ATOM_H
#define ATOM_H

#pragma once

#include <iostream>
#include <mpi.h>

namespace ATOM {

template<class T>
class atom {
public:
    explicit atom(T x = 0, T y = 0, T z = 0, T mass = 1.0);
    virtual ~atom() = default;

    // Getters
    T getX() const;
    T getY() const;
    T getZ() const;
    T getMass() const;
    T getVx() const;
    T getVy() const;
    T getVz() const;

    // Setters
    void setX(T x);
    void setY(T y);
    void setZ(T z);
    void setMass(T mass);
    void setVx(T vx);
    void setVy(T vy);
    void setVz(T vz);

    virtual void printPosition() const;

    // MPI pack and unpack methods
    void pack(MPI_Comm comm, void *outbuf, int outsize, int *position) const;
    void unpack(MPI_Comm comm, void *inbuf, int insize, int *position);

    // Calculate buffer size needed for packing
    static int bufferSize(MPI_Comm comm);

private:
    T x, y, z;
    T mass;
    T vx, vy, vz;
};

} // namespace ATOM

#include "atom.cpp" // Include the implementation file

#endif // ATOM_H
