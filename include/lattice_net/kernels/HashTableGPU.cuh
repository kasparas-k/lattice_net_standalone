#pragma once

#if !defined(__CUDACC_RTC__)
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <cuda_runtime_api.h>
    #include "device_launch_parameters.h" //needed for threadIdx and blockDim 
#endif

//adapted from https://github.com/MiguelMonteiro/permutohedral_lattice/blob/master/src/PermutohedralLatticeGPU.cuh
class HashTableGPU { 
public:

    HashTableGPU(){

    }
    HashTableGPU(int capacity, int pos_dim){
        m_capacity=capacity;
        m_pos_dim=pos_dim;
    }

    int m_capacity;
    int* m_keys; // size m_capacity x m_pos_dim  of int (or should it be short as in the original implementation)
    float* m_values; // Size m_capacity x m_val_hom_dim  of float  Stores homgeneous values, hence the m_val_hom_dim
    int* m_entries; // size m_capacity x 1 of int  entries of the matrix for recording where the splatting happened for each point. The hash value h of the key is used to index into this tensor. the result is an index that points into the rows of the values and keys tensor where the corresponding key is stored
    int* m_nr_filled; // 1x1 tensor of int storing the nr of filled cells of the keys and values tensor
    int m_pos_dim;

    #if defined(__CUDACC_RTC__)

    //cuda kernels 
    __device__ unsigned int hash(int *key) {
        unsigned int k = 0;
        for (int i = 0; i < m_pos_dim; i++) {
            k += key[i];
            k = k * 2531011;
        }
        // printf("k is %d \n", k);
        return k;
    }

    __device__ int modHash(unsigned int n){
        return(n % m_capacity);
    }

    inline __device__ void acquire( int* e ){
        while ( atomicCAS( e, -1, -2 ) +1 ); //the entires start at empty(-1) and if we succesfully change the value to locked(-2), then the old_value would be -1 and +1 would make it zero, breaking the loop
    }


    __device__ int insert(int *key ) {
        while(1){
            //to lock a certain positions
            int *e = m_entries + h;
            bool leaveLoop = false;
            while(!leaveLoop) {
                int contents = atomicCAS(e, -1, -2);;
                if(contents == -1) { //succesfuly locked it 
                    leaveLoop = true;
                    __threadfence();
                    // critical section

                    int old_filled=atomicAdd( m_nr_filled , 1);
                    for (int i = 0; i < m_pos_dim; i++) {
                        m_keys[old_filled * m_pos_dim + i] = key[i];
                    }


                    __threadfence();
                    atomicExch(e, old_filled);
                    return h;
                }else if(contents>=0){ //it has already a key inside, check if it's the same
                    leaveLoop = true; //if we match the key we would return the h, and if we don't then we leave the loop and go and check another position
                    // The cell is unlocked and has a key in it, check if it matches
                    bool match = true;
                    for (int i = 0; i < m_pos_dim && match; i++) {
                        match = (m_keys[contents*m_pos_dim+i] == key[i]);
                    }
                    if (match){
                        return h;
                    }
                }
                __threadfence();
            }

            //we left checking this position and we check the next one
            // nr_probes++;
            h++; //linear probing
            if (h >= m_capacity){
                h = 0;
            }
        }




    }

    __device__ int retrieve(int *key) {

        int h = modHash(hash(key));
        int nr_conflicts=0; //nr of times it tried to insert in a entry location but it was already used.
        int max_nr_conflicts=300;
        while (1 && nr_conflicts < max_nr_conflicts) {
            int *e = m_entries + h;

            if (*e == -1)
                return -1;

            bool match = true;
            for (int i = 0; i < m_pos_dim && match; i++) {
                match = (m_keys[(*e)*m_pos_dim+i] == key[i]);
            }
            if (match)
                return *e;

            nr_conflicts++;
            h++; //linear probing
            // h+=nr_conflicts*nr_conflicts; //quadratic probing
            if (h >= m_capacity)
                h = 0;
        }

        //if we got out of the loop it mean we failed to retreive (we had to many conflicts)
        return -1;

    }

    #endif


   
};






