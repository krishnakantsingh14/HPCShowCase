#include<mpi.h>
#include<iostream>
#include<vector> 
#include<array>

using std::cout;
using std::endl;

void create_new_mpi_type(double *a_p, double *b_p, int* n_p, MPI_Datatype *mpi_type_p) {
    std::array<int, 3> blocklengths {1,1,1}; 
    std::array<MPI_Datatype, 3> types {MPI_DOUBLE, MPI_DOUBLE,MPI_INT}; 
    MPI_Aint a_dis, b_dis, n_dis;

    MPI_Get_address(a_p, &a_dis);
    MPI_Get_address(b_p, &b_dis);
    MPI_Get_address(n_p, &n_dis);

    std::array<MPI_Aint, 3> displacement {0 , b_dis-a_dis, n_dis-a_dis};
    MPI_Type_create_struct(
        3, blocklengths.data(), displacement.data(), types.data(), mpi_type_p
    );
    
    MPI_Type_commit(mpi_type_p); 
  
}
template<typename T>
void BroadCasting(MPI_Comm &comm, MPI_Datatype &mpi_type, T &a, int rank) {
    if (rank==0) {
        a=89.0;
    }
    MPI_Bcast(&a, 1, mpi_type, 0, MPI_COMM_WORLD);

}


int main(int argc, char** argv) {
    
    MPI_Init(&argc, &argv); 

    int rank, size; 
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    MPI_Datatype mpi_type;
    double a {0.0};
    double b {2.1};
    int n {10}; 
    create_new_mpi_type(&a, &b, &n, &mpi_type);
    /*if (rank==0) {
        a = 90.0;
        b = 10.0;
        n = 100; 

    }
    MPI_Bcast( &a, 1, mpi_type, 0, MPI_COMM_WORLD);
*/
    BroadCasting(comm, mpi_type, a, rank);

    if (rank==1) {
        cout << "a = " << a << endl;
        cout << "b = " << b << endl;
        cout << "n = " << n << endl;

    }


    MPI_Type_free(&mpi_type);
    MPI_Finalize(); 
    return 0; 



}

